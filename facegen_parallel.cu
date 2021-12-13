#include <cuda_runtime.h>
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

//gpu_mem ptrs for network, inputs, and outputs
static int NETWORK_SIZE_IN_BYTES = 20549132;
static float* gpu_mem_network;
/*
Since there is no way to bring num_to_gen from main.c, 
we need to calculate each input and output separately.
(Not necessarily one at a time..)
*/
static float* gpu_mem_input;  
static float* gpu_mem_output;


void facegen_init() {
  /*
   * TODO
   * Initialize required CUDA objects. For example,
   * cudaMalloc(...)
   */
	CHECK_CUDA(cudaMalloc(&gpu_mem_network, NETWORK_SIZE_IN_BYTES * sizeof(float));
	CHECK_CUDA(cudaMalloc(&gpu_mem_input,100 * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&gpu_mem_output,64*64*3 * sizeof(float)));
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
	float *proj_w = network; network += 100 * 8192;
  float *proj_b = network; network += 8192;
  float *bn0_beta = network; network += 512;
  float *bn0_gamma = network; network += 512;
  float *bn0_mean = network; network += 512;
  float *bn0_var = network; network += 512;
  float *tconv1_w = network; network += 5 * 5 * 256 * 512;
  float *tconv1_b = network; network += 256;
  float *bn1_beta = network; network += 256;
  float *bn1_gamma = network; network += 256;
  float *bn1_mean = network; network += 256;
  float *bn1_var = network; network += 256;
  float *tconv2_w = network; network += 5 * 5 * 128 * 256;
  float *tconv2_b = network; network += 128;
  float *bn2_beta = network; network += 128;
  float *bn2_gamma = network; network += 128;
  float *bn2_mean = network; network += 128;
  float *bn2_var = network; network += 128;
  float *tconv3_w = network; network += 5 * 5 * 64 * 128;
  float *tconv3_b = network; network += 64;
  float *bn3_beta = network; network += 64;
  float *bn3_gamma = network; network += 64;
  float *bn3_mean = network; network += 64;
  float *bn3_var = network; network += 64;
  float *tconv4_w = network; network += 5 * 5 * 3 * 64;
  float *tconv4_b = network; network += 3;

  // intermediate buffer for feature maps
  float *fm0 = (float*)malloc(4 * 4 * 512 * sizeof(float));
  float *fm1 = (float*)malloc(8 * 8 * 256 * sizeof(float));
  float *fm2 = (float*)malloc(16 * 16 * 128 * sizeof(float));
  float *fm3 = (float*)malloc(32 * 32 * 64 * sizeof(float));


}

void facegen_fin() {
  /*
   * TODO
   * Finalize required CUDA objects. For example,
   * cudaFree(...)
   */
	CHECK_CUDA(gpu_mem_network);
	CHECK_CUDA(gpu_mem_input);
	CHECK_CUDA(gpu_mem_output);

}
