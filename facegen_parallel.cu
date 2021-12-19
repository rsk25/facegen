#include <cuda_runtime.h>
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
static int NETWORK_SIZE_IN_BYTES = 20549132;
extern int num_to_gen;

//gpu_mem ptrs for network, inputs, and outputs

static float* gpu_network_full;
static float* gpu_inputs;  
static float* gpu_outputs;

static float* gpu_fm0;
static float* gpu_fm1;
static float* gpu_fm2;
static float* gpu_fm3;

void facegen_init()
{
  /*
   * TODO
   * Initialize required CUDA objects. For example,
   * cudaMalloc(...)
   */
	CHECK_CUDA(cudaMalloc((void **)&gpu_network_full, NETWORK_SIZE_IN_BYTES));
	CHECK_CUDA(cudaMalloc(&gpu_inputs, num_to_gen * 100 * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&gpu_outputs, num_to_gen * 64 * 64 * 3 * sizeof(float)));

	CHECK_CUDA(cudaMalloc(&gpu_fm0, 4 * 4 * 512 * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&gpu_fm1, 8 * 8 * 256 * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&gpu_fm2, 16 * 16 * 128 * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&gpu_fm3, 32 * 32 * 64 * sizeof(float)));
		
	CHECK_CUDA(cudaDeviceSynchronize());
}

// data-parallelism w.r.t K (col. dim of output of proj)
__global__ void proj(float *in, float *out, float *weight, float *bias, int C, int K)
{
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

	CHECK_CUDA(cudaMemcpy(gpu_network_full, network, NETWORK_SIZE_IN_BYTES, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(gpu_inputs, inputs, num_to_gen * 100 * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(gpu_outputs, outputs, num_to_gen * 64 * 64 * 3 * sizeof(float), cudaMemcpyHostToDevice));
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

	dim3 gridDim(NETWORK_SIZE_IN_BYTES/64);
	dim3 blockDim(64);

	for (int n = 0; n < num_to_gen; n++){
		
		float *input = &gpu_inputs[n * 100];
		float *output = &gpu_outputs[n * 64 * 64 * 3];
		proj<<<gridDim, blockDim>>>(input, gpu_fm0, proj_w, proj_b, 100, 8192);
		batch_norm<<<gridDim, blockDim>>>(gpu_fm0, bn0_beta, bn0_gamma, bn0_mean, bn0_var, 4 * 4, 512);
		relu<<<gridDim, blockDim>>>(gpu_fm0, 4 * 4 * 512);

		tconv<<<gridDim, blockDim>>>(gpu_fm0, gpu_fm1, tconv1_w, tconv1_b, 4, 4, 512, 256);
		batch_norm<<<gridDim, blockDim>>>(gpu_fm1, bn1_beta, bn1_gamma, bn1_mean, bn1_var, 8 * 8, 256);
		relu<<<gridDim, blockDim>>>(gpu_fm1, 8 * 8 * 256);

		tconv<<<gridDim, blockDim>>>(gpu_fm1, gpu_fm2, tconv2_w, tconv2_b, 8, 8, 256, 128);
		batch_norm<<<gridDim, blockDim>>>(gpu_fm2, bn2_beta, bn2_gamma, bn2_mean, bn2_var, 16 * 16, 128);
		relu<<<gridDim, blockDim>>>(gpu_fm2, 16 * 16 * 128);

		tconv<<<gridDim, blockDim>>>(gpu_fm2, gpu_fm3, tconv3_w, tconv3_b, 16, 16, 128, 64);
		batch_norm<<<gridDim, blockDim>>>(gpu_fm3, bn3_beta, bn3_gamma, bn3_mean, bn3_var, 32 * 32, 64);
		relu<<<gridDim, blockDim>>>(gpu_fm3, 32 * 32 * 64);

		tconv<<<gridDim, blockDim>>>(gpu_fm3, output, tconv4_w, tconv4_b, 32, 32, 64, 3);
		tanh_layer<<<gridDim, blockDim>>>(output, 64 * 64 * 3);
		CHECK_CUDA(cudaDeviceSynchronize());

	}
	CHECK_CUDA(cudaMemcpy(outputs, gpu_outputs, num_to_gen * 64 * 64 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaDeviceSynchronize());
	
}

void facegen_fin() {
  /*
   * TODO
   * Finalize required CUDA objects. For example,
   * cudaFree(...)
   */
	CHECK_CUDA(cudaFree(gpu_network_full));
	CHECK_CUDA(cudaFree(gpu_inputs));
	CHECK_CUDA(cudaFree(gpu_outputs));

	CHECK_CUDA(cudaFree(gpu_fm0));
	CHECK_CUDA(cudaFree(gpu_fm1));
	CHECK_CUDA(cudaFree(gpu_fm2));
	CHECK_CUDA(cudaFree(gpu_fm3));

}
