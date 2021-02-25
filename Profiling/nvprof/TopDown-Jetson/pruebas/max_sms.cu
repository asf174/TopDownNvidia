#include <stdio.h>
#include <nppdefs.h>
int main()
{
    int value, value2;
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&value2, cudaDevAttrMaxBlocksPerMultiprocessor, device);
    
    printf("%d, %d\n", value, value2);
}