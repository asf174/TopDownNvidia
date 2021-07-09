/*
 * Program to show the COMPUTE CAPABILITY of the current device
 *
 * Author: Alvaro Saiz (UC)
 * Version: July-2021
*/
#include <stdio.h>
//#define CURRENT_DEVICE 1
#define EXIT_SUCCESSFULLY 0
#define EXIT_ERROR -1


int main(int argc, char** argv) 
{
    cudaError_t result;
    int device, blocksPerSM;
    
    //cudaSetDevice(CURRENT_DEVICE);
    cudaGetDevice(&device);
    result = cudaDeviceGetAttribute(&blocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor, device);
    if (result != cudaSuccess)
        return EXIT_ERROR;
    printf("%d\n", blocksPerSM);
    return EXIT_SUCCESSFULLY;
}