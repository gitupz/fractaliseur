#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <complex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#define DEFAULT_COLOR_DEPTH 256
#define DEFAULT_COLOR_MODE 3
#define DEFAULT_WIDTH 1920
#define DEFAULT_HEIGHT 1080
#define DEFAULT_NB_CYCLES 25

#define PI 3.141592653589793

#define MAX_THREAD 128

struct BMPheader
{
	// file header
	uint32_t file_size;
	uint32_t reserved;
	uint32_t bitmap_offset;
	// bitmap header
	uint32_t header_size;
	uint32_t bmp_width;
	uint32_t bmp_height;
	uint16_t planes;
	uint16_t bits_per_pixel;
	uint32_t compression;
	uint32_t size_bitmap;
	uint32_t horiz_resolution;
	uint32_t vert_resolution;
	uint32_t colors_used;
	uint32_t colors_important;
};

__device__ unsigned int width_d;
__device__ unsigned int height_d;
__device__ unsigned int colorDepth_d;
__device__ unsigned int nbCycles_d;
__device__ cuDoubleComplex a_d;
__device__ cuDoubleComplex b_d;

__global__ void computePixel(unsigned char* r_d)
{
	int i,j,x,y;
	cuDoubleComplex z_d, c_d;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	x = (i % width_d) - width_d / 2;
	y = (i / width_d) - height_d / 2;
	c_d = make_cuDoubleComplex((double)x,(double)y);
    c_d = cuCadd(cuCmul(a_d, c_d), b_d);
	z_d = make_cuDoubleComplex(0.0,0.0);
	for(j = 0; j < nbCycles_d && cuCabs(z_d) < 2; j++)
	{
		z_d = cuCadd(cuCmul(z_d, z_d), c_d);
	}
	if(cuCabs(z_d) > 2)
	{
		r_d[i*3] = j * colorDepth_d / nbCycles_d;
		r_d[i*3+1] = r_d[i*3];
		r_d[i*3+2] = r_d[i*3];
	}
	else
	{
		r_d[i*3] = 255;
		r_d[i*3+1] = 255;
		r_d[i*3+2] = 255;
	}
}

__global__ void setParameters(unsigned int w, unsigned int h,
		unsigned int cd, unsigned int n, cuDoubleComplex a,
		cuDoubleComplex b)
{
	width_d = w;
	height_d = h;
	colorDepth_d = cd;
	nbCycles_d = n;
	a_d = a;
	b_d = b;
}

int writeBMP(char* filename, unsigned int width, unsigned int height,
		unsigned int colorDepth, unsigned char* data, unsigned int size)

{
	struct BMPheader header;
	// fulfilling file header
	header.file_size = 54 + size;
	header.reserved = 0;
	header.bitmap_offset = 54;
	header.header_size = 40;
	header.bmp_width = width;
	header.bmp_height = height;
	header.planes = 1;
	header.bits_per_pixel = 24;
	header.compression = 0;
	header.size_bitmap = size;
	header.horiz_resolution = 0;
	header.vert_resolution = 0;
	header.colors_used = 0;
	header.colors_important = 0;
		
	// writing result in a file
    FILE* file = fopen(filename, "wb");
    if(file == NULL)
    {
		return -1;
    }
    fputs("BM", file);
    fwrite(&header, sizeof(struct BMPheader), 1, file);
    fwrite(data, sizeof(unsigned char), size, file);
    fclose(file);
    
    return 0;
}

int createImage(char* filename, unsigned int width, unsigned int height,
		unsigned int colorDepth, unsigned int nbCycles,
		cuDoubleComplex a, cuDoubleComplex b)
{
	// initializations
	struct timespec start, end;
	double time_elapsed;
	unsigned char* r;
	unsigned char* r_device;
	
	// starting time measure
	clock_gettime(CLOCK_MONOTONIC, &start);
	
	// allocating memory on RAM
	r = (unsigned char*) malloc(sizeof(unsigned char) * width * height * DEFAULT_COLOR_MODE);
	
	// allocating memory on GPU
    cudaMalloc((void**) &r_device, sizeof(unsigned char) * width * height * DEFAULT_COLOR_MODE);

	// setting scene parameters on the GPU
	setParameters<<<1,1>>>(width, height, colorDepth, nbCycles, a, b);
    
    // making the GPU do the job
    computePixel<<<width * height / MAX_THREAD, MAX_THREAD>>>(r_device);

    // copying result from GPU to RAM
    cudaMemcpy(r, r_device, sizeof(unsigned char) * width * height * DEFAULT_COLOR_MODE, cudaMemcpyDeviceToHost);
    
    // writing the result in a bitmap
    writeBMP(filename, width, height, DEFAULT_COLOR_MODE, r,
			sizeof(unsigned char) * width * height * DEFAULT_COLOR_MODE);
    
    // freeing GPU's and CPU's memory
    cudaFree(r_device);
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

int main(int argc, char* argv[])
{
	int n, nbImage = 1000;
	double complex a0, b0;
	cuDoubleComplex a, b;
	char imageName[256];
	for(n = 0; n < nbImage; n++)
	{
		a0 = 3.0 / DEFAULT_WIDTH * cexp(3.0 * 2 * PI * n / nbImage * I);
		b0 = 0;
		printf("%g + %gi\n", creal(a0), cimag(a0));
		a = make_cuDoubleComplex(creal(a0),cimag(a0));
		b = make_cuDoubleComplex(creal(b0),cimag(b0));
		snprintf(imageName, sizeof(imageName), "seq%05d.bmp", n);
		createImage(imageName, DEFAULT_WIDTH, DEFAULT_HEIGHT,
				DEFAULT_COLOR_DEPTH, DEFAULT_NB_CYCLES, a, b);
	} 
}
