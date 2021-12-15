#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "facegen.h"


#define CHECK_CUDA(err) \
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
extern num_to_gen;

//gpu_mem ptrs for network, inputs, and outputs
static int NETWORK_SIZE_IN_BYTES = 20549132;
static float* gpu_network;
static float* gpu_inputs;  
static float* gpu_outputs;


void facegen_init() {
  /*
   * TODO
   * Initialize required CUDA objects. For example,
   * cudaMalloc(...)
   */
	CHECK_CUDA(cudaMalloc(&gpu_network, NETWORK_SIZE_IN_BYTES * sizeof(float));
	CHECK_CUDA(cudaMalloc(&gpu_input,num_to_gen * 100 * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&gpu_output,num_to_gen * 64*64*3 * sizeof(float)));
}

// data-parallelism w.r.t K (col. dim of output of proj)
__global__ void proj(float *in, float *out, float *weight, float *bias, int C, K){
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	if (k >= K) return;
	
	float s = 0;
	for (int c = 0; c<C; c++){
		s += in[c] * weight[c]
	}
	s += bias[k];
	out[k] = s;
}

__global__ void batch_norm(float *inout, float *beta, float *gamma, float *mean, float *var, int HW, int C){		
	int hw = blockDim.x * blockIdx.x + threadIdx.x;
	if (hw >= HW) return;
	
	for (int c = 0; c < C; c++){
		float scaled_gamma = gamma[c] / sqrtf(var[c] + 1e-5);
		inout[hw * C + c] = scaled_gammma * inout[hw * C + c] + (beta[c] - scaled_gammma * mean[c]);
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
	
	for (int n = 0; n < num_to_gen; n++){
		
		/* Add MPI_Send, MPI_Recv here*/ 

		// Linear projection layer
		dim3 gridDim();
		dim3 blockDim();
		proj<<<>>>(gpu_input, gpu_fm0, proj_w, proj_b, 100, 8192);
		batch_norm<<<>>>(gpu_input, gpu_fm0, proj_w, proj_b, 100, 8192);
		relu<<<>>>(gpu_input, gpu_fm0, proj_w, proj_b, 100, 8192);

		
	}


}

void facegen_fin() {
  /*
   * TODO
   * Finalize required CUDA objects. For example,
   * cudaFree(...)
   */
	CHECK_CUDA(gpu_network);
	CHECK_CUDA(gpu_input);
	CHECK_CUDA(gpu_output);

}
