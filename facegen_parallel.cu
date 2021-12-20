#include <cuda_runtime.h>
#include <mpi.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "facegen.h" 

#define CHECK_CUDA(err)  \
  do { \
    cudaError_t CHECK_CUDA_err = (err); \
    if (CHECK_CUDA_err != cudaSuccess) { \
      printf("[%s:%d] CUDA error %d (%s)\n", __FILE__, __LINE__, CHECK_CUDA_err, cudaGetErrorString(CHECK_CUDA_err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)


/*
 * TODO
 * Define global variables here.
 */

extern const int NETWORK_SIZE_IN_BYTES;
extern int num_to_gen;
extern int per_node_size;
extern int tag;

//gpu_mem ptrs for network, inputs, and outputs

static float* gpu_network_full;
static float* gpu_inputs[2];  
static float* gpu_outputs[2];

static float* gpu_fm0;
static float* gpu_fm1;
static float* gpu_fm2;
static float* gpu_fm3;

static int mpi_rank; 
static int mpi_size;
static MPI_Request request;
static MPI_Request* nodeRequests;
static MPI_Status status;
static MPI_Status* nodeStatus;

static cudaStream_t stream[3];


void facegen_init(){
  /*
   * TODO
   * Initialize required CUDA objects. For example,
   * cudaMalloc(...)
   */
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	int num_of_device;
	cudaGetDeviceCount(&num_of_device);
	cudaSetDevice(mpi_rank % 4);

	nodeStatus = (MPI_Status *)malloc(mpi_size * sizeof(MPI_Status));
	nodeRequests = (MPI_Request *)malloc(mpi_size * sizeof(MPI_Request));

	if (mpi_rank == 0){
		per_node_size = num_to_gen - ((mpi_size - 1) * per_node_size);
	}

	CHECK_CUDA(cudaMalloc(&gpu_network_full, NETWORK_SIZE_IN_BYTES));
	// double buffering
	for (int i = 0; i < 2; i++) {
		CHECK_CUDA(cudaMalloc(&gpu_inputs[i], 100 * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&gpu_outputs[i], 64 * 64 * 3 * sizeof(float)));
	}
	CHECK_CUDA(cudaMalloc(&gpu_fm0, 4 * 4 * 512 * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&gpu_fm1, 8 * 8 * 256 * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&gpu_fm2, 16 * 16 * 128 * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&gpu_fm3, 32 * 32 * 64 * sizeof(float)));
	
	// 4 gpus --> 3 + 1 streams
	for (int i = 0; i < 3; i++){
		CHECK_CUDA(cudaStreamCreate(&stream[i]));
		CHECK_CUDA(cudaStreamSynchronize(stream[i]));
	}
	CHECK_CUDA(cudaDeviceSynchronize());

}

// data-parallelism w.r.t K (col. dim of output of proj)
__global__ void proj(float *in, float *out, float *weight, float *bias, int C, int K){
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	if (k >= K) return;
	
	float s = 0;
	for (int c = 0; c < C; c++){
		s += in[c] * weight[c * K + k];
	}
	s += bias[k];
	out[k] = s;
}

__global__ void batch_norm(float *inout, float *beta, float *gamma, float *mean, float *var, int HW, int C){		
	int hw = blockDim.x * blockIdx.x + threadIdx.x;
	if (hw >= HW) return;
	
	for (int c = 0; c < C; c++){
		float scaled_gamma = gamma[c] / sqrtf(var[c] + 1e-5);
		inout[hw * C + c] = scaled_gamma * inout[hw * C + c] + (beta[c] - scaled_gamma * mean[c]);
	}
}

__global__ void tanh_layer(float *inout, int HWC){
	int hwc = blockDim.x * blockIdx.x + threadIdx.x;
	if (hwc >= HWC) return;

	inout[hwc] = tanhf(inout[hwc]);
}

__global__ void relu(float *inout, int HWC){
	int hwc = blockDim.x * blockIdx.x + threadIdx.x;
	if (hwc >= HWC) return;

	inout[hwc] = fmaxf(inout[hwc], 0);
}

__global__ void tconv(float *in, float *out, float *weight, float* bias, int H_IN, int W_IN, int C, int K){
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	if (k >= K) return;

	int H_OUT = H_IN * 2;
 	int W_OUT = W_IN * 2;
	for (int h_out = 0; h_out < H_OUT; h_out++) {
		for (int w_out = 0; w_out < W_OUT; w_out++) {
			float ss = 0;
			for (int r = 0; r < 5; r++) {
				for (int s = 0; s < 5; s++) {
					// top and left side has padding 3, bottom and right side has padding 2
					// so subtract 3
					int h_in = h_out - 3 + r;
					int w_in = w_out - 3 + s;
					// stride is 2, so check coordinates fall into input element or empty space
					if (h_in % 2 == 0 && w_in % 2 == 0) {
						h_in /= 2;
						w_in /= 2;
						// boundary check
						if (0 <= h_in && h_in < H_IN && 0 <= w_in && w_in < W_IN) {
							for (int c = 0; c < C; c++) {
								// filter is stored in reverse; so use [4 - r][4 - s] instead of [r][s]
								// ss += in[h_in][w_in][c] * weight[4 - r][4 - s][k][c];
								ss += in[(h_in * W_IN + w_in) * C + c] * weight[(((4 - r) * 5 + (4 - s)) * K + k) * C + c];
							}
						}
					}
				}
			}
			ss += bias[k];
			// out[h_out][w_out][k] = ss;
			out[(h_out * W_OUT + w_out) * K + k] = ss;
		}
	}
}


void facegen(int num_to_gen, float *network, float *inputs, float *outputs) {
  /*
   * TODO
   * Implement facegen computation here.
   * See "facegen_seq.c" if you don't know what to do.
   *
   * Below functions should be implemented in here:
   * Host-to-devie memory copy,
   * CUDA kernel launch,
   * Device-to-host memory copy
   */

	if (mpi_rank == 0){
		int offset = 100 * (num_to_gen / mpi_size);
		float* nodeInputs = inputs + 100 * per_node_size;
		for (int i = 1; i < mpi_size; i++){
			MPI_Isend(network, NETWORK_SIZE_IN_BYTES / sizeof(float), MPI_FLOAT, i, tag, MPI_COMM_WORLD, &request);
			MPI_Isend(nodeInputs + (i-1)*offset, offset, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, &status);
		}
	} else {
		int offset = 100 * (num_to_gen / mpi_size);
		for (int i = 1; i < mpi_size; i++){
			MPI_Irecv(network, NETWORK_SIZE_IN_BYTES / sizeof(float), MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &nodeRequests[i]);
			MPI_Irecv(inputs, offset, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &nodeRequests[i]);
			MPI_Wait(&nodeRequests[i], &nodeStatus[i]);
		}
	}

	CHECK_CUDA(cudaMemcpy(gpu_network_full, network, NETWORK_SIZE_IN_BYTES, cudaMemcpyHostToDevice));
	float* gpu_network = gpu_network_full;

	float *proj_w = gpu_network; gpu_network += 100 * 8192;
  float *proj_b = gpu_network; gpu_network += 8192;
  float *bn0_beta = gpu_network; gpu_network += 512;
  float *bn0_gamma = gpu_network; gpu_network += 512;
  float *bn0_mean = gpu_network; gpu_network += 512;
  float *bn0_var = gpu_network; gpu_network += 512;
  float *tconv1_w = gpu_network; gpu_network += 5 * 5 * 256 * 512;
  float *tconv1_b = gpu_network; gpu_network += 256;
  float *bn1_beta = gpu_network; gpu_network += 256;
  float *bn1_gamma = gpu_network; gpu_network += 256;
  float *bn1_mean = gpu_network; gpu_network += 256;
  float *bn1_var = gpu_network; gpu_network += 256;
  float *tconv2_w = gpu_network; gpu_network += 5 * 5 * 128 * 256;
  float *tconv2_b = gpu_network; gpu_network += 128;
  float *bn2_beta = gpu_network; gpu_network += 128;
  float *bn2_gamma = gpu_network; gpu_network += 128;
  float *bn2_mean = gpu_network; gpu_network += 128;
  float *bn2_var = gpu_network; gpu_network += 128;
  float *tconv3_w = gpu_network; gpu_network += 5 * 5 * 64 * 128;
  float *tconv3_b = gpu_network; gpu_network += 64;
  float *bn3_beta = gpu_network; gpu_network += 64;
  float *bn3_gamma = gpu_network; gpu_network += 64;
  float *bn3_mean = gpu_network; gpu_network += 64;
  float *bn3_var = gpu_network; gpu_network += 64;
  float *tconv4_w = gpu_network; gpu_network += 5 * 5 * 3 * 64;
  float *tconv4_b = gpu_network; gpu_network += 3;
	
	/* Add MPI_Send, MPI_Recv here*/ 

	dim3 gridDim(8192);
	dim3 blockDim(64);

		
	for (int n = 0; n < per_node_size; n++){
		
		// per each buffer
		int idx = n % 2;
		float *input = gpu_inputs[idx];
		float *output = gpu_outputs[idx];
		CHECK_CUDA(cudaMemcpyAsync(gpu_inputs[idx], &inputs[n * 100], 100 * sizeof(float), cudaMemcpyHostToDevice, stream[idx]));

		proj<<<gridDim, blockDim, 0, stream[idx]>>>(input, gpu_fm0, proj_w, proj_b, 100, 8192);
		batch_norm<<<gridDim, blockDim, 0, stream[idx]>>>(gpu_fm0, bn0_beta, bn0_gamma, bn0_mean, bn0_var, 4 * 4, 512);
		relu<<<gridDim, blockDim, 0, stream[idx]>>>(gpu_fm0, 4 * 4 * 512);
		
		tconv<<<gridDim, blockDim, 0, stream[idx]>>>(gpu_fm0, gpu_fm1, tconv1_w, tconv1_b, 4, 4, 512, 256);
		batch_norm<<<gridDim, blockDim, 0, stream[idx]>>>(gpu_fm1, bn1_beta, bn1_gamma, bn1_mean, bn1_var, 8 * 8, 256);
		relu<<<gridDim, blockDim, 0, stream[idx]>>>(gpu_fm1, 8 * 8 * 256);

		tconv<<<gridDim, blockDim, 0, stream[idx]>>>(gpu_fm1, gpu_fm2, tconv2_w, tconv2_b, 8, 8, 256, 128);
		batch_norm<<<gridDim, blockDim, 0, stream[idx]>>>(gpu_fm2, bn2_beta, bn2_gamma, bn2_mean, bn2_var, 16 * 16, 128);
		relu<<<gridDim, blockDim, 0, stream[idx]>>>(gpu_fm2, 16 * 16 * 128);

		tconv<<<gridDim, blockDim, 0, stream[idx]>>>(gpu_fm2, gpu_fm3, tconv3_w, tconv3_b, 16, 16, 128, 64);
		batch_norm<<<gridDim, blockDim, 0, stream[idx]>>>(gpu_fm3, bn3_beta, bn3_gamma, bn3_mean, bn3_var, 32 * 32, 64);
		relu<<<gridDim, blockDim, 0, stream[idx]>>>(gpu_fm3, 32 * 32 * 64);

		tconv<<<gridDim, blockDim, 0, stream[idx]>>>(gpu_fm3, output, tconv4_w, tconv4_b, 32, 32, 64, 3);
		tanh_layer<<<gridDim, blockDim, 0, stream[idx]>>>(output, 64 * 64 * 3);
		CHECK_CUDA(cudaDeviceSynchronize());

		CHECK_CUDA(cudaMemcpyAsync(&outputs[n * 64 * 64 * 3], gpu_outputs[idx], 64 * 64 * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream[idx]));

	}
	
	for (int i = 0; i < 3; i++){
		CHECK_CUDA(cudaStreamSynchronize(stream[i]));
	}
	CHECK_CUDA(cudaDeviceSynchronize());
	
	// recieve output from nodes
	if (mpi_rank == 0){
		int offset = 64 * 64 * 3 * (num_to_gen / mpi_size);
		float* nodeOutputs = outputs + 64 * 64 * 3 * per_node_size;
		for (int i = 1; i < mpi_size; i++){
			MPI_Irecv(nodeOutputs + (i-1)*offset, offset, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &nodeRequests[i]);
		}
		for (int i = 1; i < mpi_size; i++){
			MPI_Wait(&nodeRequests[i], &nodeStatus[i]);
		}
	} else {
		int offset = 64 * 64 * 3 * (num_to_gen / mpi_size);
		MPI_Isend(outputs, offset, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);
	}
	MPI_Barrier(MPI_COMM_WORLD);	
}

void facegen_fin() {
  /*
   * TODO
   * Finalize required CUDA objects. For example,
   * cudaFree(...)
   */
	CHECK_CUDA(cudaFree(gpu_network_full));
	for (int i = 0; i < 2; i++){
		CHECK_CUDA(cudaFree(gpu_inputs[i]));
		CHECK_CUDA(cudaFree(gpu_outputs[i]));
	}


	CHECK_CUDA(cudaFree(gpu_fm0));
	CHECK_CUDA(cudaFree(gpu_fm1));
	CHECK_CUDA(cudaFree(gpu_fm2));
	CHECK_CUDA(cudaFree(gpu_fm3));


	for (int i = 0; i < 3; i++){
		CHECK_CUDA(cudaStreamDestroy(stream[i]));
	}

}
