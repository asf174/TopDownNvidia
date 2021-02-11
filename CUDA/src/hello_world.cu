#include <stdio.h>
#include <cuda_profiler_api.h>


__global__ void printHola() {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	printf("Hola mundo desde el device con thread: %d\n",idx);
	// printf para evitar que salga warning al compilar
}

int main() {
	//printf("Hola Mundo desde el host\n");

	cudaProfilerStart();
	printHola<<<1,65>>>();
	cudaDeviceSynchronize();
	cudaProfilerStop();
	return 0;
}
