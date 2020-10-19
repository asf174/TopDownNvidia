#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 3

__global__ void add(int a, int b, int* result) {
	*result = a + b;
}

int main() {

	int* operation = (int *) malloc(N*sizeof(int));
	operation[0] = 2;
	operation[1] = 200;

	// direcciones de los operandos del DEVICE
	int *d_a, *d_b, *d_r;

	// reservo memoria en DEVICE
	cudaMalloc((void **) &d_a,sizeof(int)); 
	cudaMalloc((void **) &d_b,sizeof(int));
	cudaMalloc((void **) &d_r,sizeof(int));


	// copio la memoria del HOST en la de DEVICE
	//cudaMemcpy(destino (device), origen (host), tamanho, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, &operation[0], sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &operation[1], sizeof(int), cudaMemcpyHostToDevice);

	// llamo funcion device
	add<<<1,1>>>(operation[0],operation[1],d_r);

	cudaDeviceSynchronize();
	// paso el resultado de memoria de DEVICE  a memoria de HOST
	int result;
	cudaMemcpy(&result, d_r, sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("El resultado es %d\n",result);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_r);
	free(operation);

}
