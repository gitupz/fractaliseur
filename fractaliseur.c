#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <complex.h>

#define COLOR_DEPTH 256
#define DEFAULT_WIDTH 3840
#define DEFAULT_HEIGHT 2160


void computePixelNoGPU(double complex a, double complex b, unsigned short n, unsigned char* r)
{
	unsigned int i, j, x, y;
	double complex c, z;
	for(x=0; x<DEFAULT_WIDTH; x++)
    {
        for(y=0; y<DEFAULT_HEIGHT; y++)
        {
			i = x + y * DEFAULT_WIDTH;
			c = x + y * I;
			c = a * c + b;
            z = 0.0;
			for(j = 0; j < n && abs(z) < 2; j++)
			{
				z = z * z + c;
			}
			if(abs(z) > 2)
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


int main()
{
	// initializations
	struct timespec start, end;
	double time_elapsed;
	unsigned char* r;
	double complex a;
	double complex b;
	
	// starting time measure
	clock_gettime(CLOCK_MONOTONIC, &start);
	
	// zoom and rotation parameters
	a = 1.0 / DEFAULT_WIDTH * 3  + 0.0005 * I;
    b = -1.5 + -1.6 * I;
	
	// allocating memory on RAM
	r = (unsigned char*) malloc(sizeof(unsigned char)*DEFAULT_WIDTH*DEFAULT_HEIGHT*3);

    // making the CPU do the job
    computePixelNoGPU(a, b, 50, r);

    // writing result in a file
    FILE* file = fopen("test.data", "wb");
    if(file == NULL)
    {
		return -1;
    }
    fwrite(r, sizeof(unsigned char), DEFAULT_WIDTH*DEFAULT_HEIGHT*3, file);
    fclose(file);
    
    // freeing GPU's and CPU's memory
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
