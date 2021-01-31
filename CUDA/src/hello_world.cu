#include <stdio.h>

__global__ void printHola() {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	//printf("Hola mundo desde el device con thread: %d\n",idx);
}

int main() {
	//printf("Hola Mundo desde el host\n");

	printHola<<<1,2>>>();
	cudaDeviceSynchronize();
	return 0;
}
