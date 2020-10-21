#include <stdio.h>
#define N 1000
// 1 thread: 1.56 segundos
__global__ void addMatrix(int* a, int* b, int* result, int block)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	printf("Hola mundo desde el device con thread: %d\n",idx);
	for (int i = block*idx; i <block*(idx + 1); i++)
			result[i] = a[i] + b[i];
}

void 
printMtx(int * matrix)
{
	int i,j;
	for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%d\t", matrix[i*N+j]);
        }
        printf("\n");
    }
}

int
main(int argc, char* argv[])
{

	int *matrixA,*matrixB,*matrixResult;
	
	// create matrix
	
	for (int i = 0; i < N; i++) {
		matrixA = (int *) malloc(N * N* sizeof(int));
		matrixB = (int *) malloc(N * N* sizeof(int));
		matrixResult = (int *) malloc(N * N * sizeof(int));
	}
	for(int i = 0; i < N*N; i++) {
			matrixA[i] = 4;
			matrixB[i] = 10;
	}
	

	// allocate memory in device
	int *matrixA_d, *matrixB_d, *matrixResult_d;

	cudaMalloc((void **) &matrixA_d, N * N * sizeof(int));
	cudaMalloc((void **) &matrixB_d, N * N * sizeof(int));
	cudaMalloc((void **) &matrixResult_d, N * N * sizeof(int));

	cudaMemcpy(matrixA_d,matrixA,N * N * sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(matrixB_d,matrixB,N * N * sizeof(int),cudaMemcpyHostToDevice);
		 

	int numThreads = 15;
	addMatrix<<<1,numThreads>>>(matrixA_d,matrixB_d,matrixResult_d,N*N/numThreads);
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaDeviceSynchronize();
	
	cudaMemcpy(matrixResult,matrixResult_d,N*N*sizeof(int),cudaMemcpyDeviceToHost);

	//printMtx(matrixResult);

	cudaFree(matrixA_d);
	cudaFree(matrixB_d);
	cudaFree(matrixResult_d);

	
	
}
