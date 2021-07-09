#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <time.h>

#ifndef N 
	#define N 200
#endif

#ifndef numThreadsPerBlock
	#define numThreadsPerBlock 256
#endif

#ifndef numBlock
	#define numBlock (ceil( (float) N*N/numThreadsPerBlock))
#endif

using clock_value_t = long long;

__device__ void sleep2(clock_value_t sleep_cycles)
{
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start; } 
    while (cycles_elapsed < sleep_cycles);
}

__global__ void addMatrix(int* a, int* b, int* result, int size)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	printf("%d\n", idx);
	if (idx % 2 == 0) {
		result[idx] = a[idx] + b[idx];
	} else {
		result[idx] = 1;
	}
	/*if (result[idx] == 4)
		result[idx*result[idx] % size] = a[idx*result[idx] % size] + b[idx*result[idx] % size];
	else
		result[idx*result[idx] % size] = a[idx*result[idx] % size];*/
    //sleep2((clock_value_t) 1000000^25);
}


__global__ void addMatrix(int* a, int* b, int* result, int size, int id)
{
    if (id < 4) {
	    int idx = blockDim.x*blockIdx.x + threadIdx.x;	
	    if (idx < size)
		    result[idx] = a[idx] + b[idx];
    }
}


__global__ void addMatrix2(int* a, int* b, int* result, int size)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;	
	if (idx < size)
		result[idx] = a[idx] + b[idx];
	/*if (result[idx] == 4)
		result[idx*result[idx] % size] = a[idx*result[idx] % size] + b[idx*result[idx] % size];
	else
		result[idx*result[idx] % size] = a[idx*result[idx] % size];*/
}


// print matrix indicated by argument
void 
printMtx(int * matrix)
{
	int i,j;
	for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
            printf("%d\t", matrix[i*N+j]);
        printf("\n");
    }
}

double time() {
	struct timeval time;

  	/* take time of execution */
  	gettimeofday(&time,NULL);
	return time.tv_sec*1000.0 + time.tv_usec/1000.0;
}

int
main(int argc, char* argv[])
{
    int i = 0;
top:
    if(argc > 1)
        printf("%s", argv[1]);
    i++;
	// create events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// another way to measure time
	double initime = time();

	int *matrixA,*matrixB,*matrixResult;
	
	// create matrix
	for (int i = 0; i < N; i++) {
		matrixA = (int *) malloc(N * N * sizeof(int));
		matrixB = (int *) malloc(N * N * sizeof(int));
		matrixResult = (int *) malloc(N * N * sizeof(int));
	}
	for(int i = 0; i < N*N; i++) {
			matrixA[i] = 4;
			matrixB[i] = 10;
	}
	
	// allocate memory in device
	int *matrixA_d, *matrixB_d, *matrixResult_d;

	cudaMalloc((void **) &matrixA_d, N * N *sizeof(int));
	cudaMalloc((void **) &matrixB_d, N * N * sizeof(int));
	cudaMalloc((void **) &matrixResult_d, N * N * sizeof(int));

	cudaMemcpy(matrixA_d, matrixA, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(matrixB_d, matrixB, N * N * sizeof(int), cudaMemcpyHostToDevice);
		 
	cudaEventRecord(start);
	
	cudaProfilerStart();
	
	//for (int i = 0; i < 10000000; i++) {
	//addMatrix<<<numBlock,numThreadsPerBlock>>>(matrixA_d,matrixB_d,matrixResult_d,N*N);
	addMatrix<<<1,32>>>(matrixA_d,matrixB_d,matrixResult_d,N*N);
    //addMatrix<<<numBlock,numThreadsPerBlock>>>(matrixA_d,matrixB_d,matrixResult_d,N*N, 5);
    //addMatrix<<<numBlock,numThreadsPerBlock>>>(matrixA_d,matrixB_d,matrixResult_d,N*N);

         //addMatrix2<<<numBlock,numThreadsPerBlock>>>(matrixA_d,matrixB_d,matrixResult_d,N*N);
        //sleep(60);
    //}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaDeviceSynchronize();
	cudaProfilerStop();
	cudaEventRecord(stop);
	double endtime = time();
	
	cudaMemcpy(matrixResult,matrixResult_d,N*N*sizeof(int),cudaMemcpyDeviceToHost);

	//printMtx(matrixResult);
	//printMtx(matrixA);
	//printMtx(matrixB);
	//printMtx(matrixResult);

	cudaFree(matrixA_d);
	cudaFree(matrixB_d);
	cudaFree(matrixResult_d);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("NUMBLOCKS: %d THREADS_PER_BLOCK: %d\n", (int) numBlock, (int) numThreadsPerBlock);
	//printf("Time elapsed in DEVICE: %f milliseconds / %g seconds\n",milliseconds, milliseconds/1000);
	//printf("Time elapsed in DEVICE (%d,%d) N = %d : %g milliseconds / %g seconds\n", numBlock,numThreadsPerBlock,N,
	//endtime - initime,(endtime - initime)/1000);
	//printf("%g %d\n",endtime - initime,numThreadsPerBlock);
    //if (i < 3)
    // goto top;
    return 0;
}
