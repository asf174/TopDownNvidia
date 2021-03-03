#include <stdio.h>
//#define CURRENT_DEVICE 1
#define EXIT_SUCCESSFULLY 0
#define EXIT_ERROR -1

int main(int argc, char** argv)
{
    cudaError_t result;
    int device, freq_khz;
    
    cudaGetDevice(&device);
    result = cudaDeviceGetAttribute(&freq_khz, cudaDevAttrClockRate, device);
    if (result != cudaSuccess)
        exit(EXIT_ERROR);
    int freq_hz = (long long int) freq_khz * 1000;  // Convert from KHz.
    printf("%d\n", freq_hz);
    exit(EXIT_SUCCESSFULLY);
}  // preguntar



