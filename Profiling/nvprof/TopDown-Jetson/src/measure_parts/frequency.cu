#include <stdio.h>
//#define CURRENT_DEVICE 1
#define EXIT_SUCCESSFULLY 0
#define EXIT_ERROR -1

int main(int argc, char** argv)
{
    cudaError_t result;
    int device, frequency;

    //cudaSetDevice(CURRENT_DEVICE);
    cudaGetDevice(&device);
    result = cudaDeviceGetAttribute(&frequency, cudaDevAttrClockRate, device);
    
    if (result != cudaSuccess)
        exit(EXIT_ERROR);
    printf("%d\n",result);
    exit(EXIT_SUCCESSFULLY);
}  // preguntar



