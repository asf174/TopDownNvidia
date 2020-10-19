#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

__global__ void informacion() {
	printf("Hola mundo de parte del thread [%d,%d] desde el device\n",threadIdx.x, blockIdx.x);
}

int main() {

	// program's body
	printf("Hola, mundo!\n");
	printf("\nPulsa ENTER para finalizar...");
	fflush(stdin);
	char tecla = getchar();
	informacion<<<1,1>>>();
	cudaDeviceSynchronize(); 
	return 0;
}
