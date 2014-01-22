// to use CUDA, uncomment the following line
#define USE_CUDA

#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#define COLOR_DEPTH 256
#define DEFAULT_WIDTH 3840
#define DEFAULT_HEIGHT 2160
#define MAX_THREAD 65536


#ifndef USE_CUDA
void computePixelNoGPU(cuDoubleComplex a, cuDoubleComplex b, unsigned short n, unsigned char* r)
{
        int i, j;
        cuDoubleComplex c, z;
        for(unsigned int x=0; x<DEFAULT_WIDTH; x++)
    {
        for(unsigned int y=0; y<DEFAULT_HEIGHT; y++)
        {
                        i = x + y * DEFAULT_WIDTH;
                        c = make_cuDoubleComplex((double)x,(double)y);
                        c = cuCadd(cuCmul(a, c), b);
            z = make_cuDoubleComplex(0.0,0.0);
                        for(j = 0; j < n && cuCabs(z) < 2; j++)
                        {
                                z = cuCadd(cuCmul(z, z), c);
                        }
                        if(cuCabs(z) > 2)
                        {
                                r[i*3] = j * COLOR_DEPTH / n;
                                r[i*3+1] = r[i*3];
                                r[i*3+2] = r[i*3];
                        }
                        else
                        {
                                r[i*3] = 0;
                                r[i*3+1] = 0;
                                r[i*3+2] = 0;
                        }
         }
    }
}
#else
__global__ void computePixel(cuDoubleComplex* a_d, cuDoubleComplex* b_d, unsigned short n, unsigned char* r_d)
{
        int i,j,x,y;
        cuDoubleComplex z_d, c_d;
        i = blockIdx.x * blockDim.x + threadIdx.x;
        x = i % DEFAULT_WIDTH;
        y = i / DEFAULT_WIDTH;
        c_d = make_cuDoubleComplex((double)x,(double)y);
    c_d = cuCadd(cuCmul(a_d[0], c_d), b_d[0]);
        z_d = make_cuDoubleComplex(0.0,0.0);
        for(j = 0; j < n && cuCabs(z_d) < 2; j++)
        {
                z_d = cuCadd(cuCmul(z_d, z_d), c_d);
        }
        if(cuCabs(z_d) > 2)
        {
                r_d[i*3] = j * COLOR_DEPTH / n;
                r_d[i*3+1] = r_d[i*3];
                r_d[i*3+2] = r_d[i*3];
        }
        else
        {
                r_d[i*3] = 0;
                r_d[i*3+1] = 0;
                r_d[i*3+2] = 0;
        }
}
#endif


int main()
{
	// initializations
	struct timespec start, end;
	double time_elapsed;
	unsigned char* r;
	cuDoubleComplex a;
	cuDoubleComplex b;
#ifdef USE_CUDA
	unsigned char* r_device;
	cuDoubleComplex* a_device;
	cuDoubleComplex* b_device;
#endif
        
	// starting time measure
	clock_gettime(CLOCK_MONOTONIC, &start);
	
	// zoom and rotation parameters
	a = make_cuDoubleComplex(1.0/DEFAULT_WIDTH*3,0.0005);
    b = make_cuDoubleComplex(-1.5,-1.6);
        
	// allocating memory on RAM
	r = (unsigned char*) malloc(sizeof(unsigned char)*DEFAULT_WIDTH*DEFAULT_HEIGHT*3);
        
#ifdef USE_CUDA
	// allocating memory on GPU
    cudaMalloc((void**) &r_device, sizeof(unsigned char)*DEFAULT_WIDTH*DEFAULT_HEIGHT*3);
    cudaMalloc((void**) &a_device, sizeof(cuDoubleComplex));
    cudaMalloc((void**) &b_device, sizeof(cuDoubleComplex));
    
    // copy a and b
    cudaMemcpy(a_device, &a, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, &b, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    

    // making the GPU do the job
    computePixel<<<DEFAULT_WIDTH*DEFAULT_HEIGHT/MAX_THREAD, MAX_THREAD>>>(a_device, b_device, 256, r_device);

    // copying result from GPU to RAM
    cudaMemcpy(r, r_device, sizeof(unsigned char)*DEFAULT_WIDTH*DEFAULT_HEIGHT*3, cudaMemcpyDeviceToHost);
#else
    // making the CPU do the job
    computePixelNoGPU(a, b, 256, r);
#endif
    
    // writing result in a file
    FILE* file = fopen("test.data", "wb");
    if(file == NULL)
    {
                return -1;
    }
    fwrite(r, sizeof(unsigned char), DEFAULT_WIDTH*DEFAULT_HEIGHT*3, file);
    fclose(file);
    
    // freeing GPU's and CPU's memory
#ifdef USE_CUDA
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(r_device);
#endif
    free(r);
    
    // ending the timer and print the result
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_elapsed = end.tv_nsec - start.tv_nsec;
    time_elapsed /= 1000000000.0;
    time_elapsed += (double) (end.tv_sec - start.tv_sec);
    printf("time elapsed : %f\n", time_elapsed);
    
    // return success state
    return 0;
}
