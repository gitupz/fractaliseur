# définition des cibles particulières
.PHONY: clean, mrproper
 
# désactivation des règles implicites
.SUFFIXES:
 
# définition des variables
CC = gcc
CFLAGS = -W -Wall -v
NVCC = nvcc
NVFLAGS = -arch=compute_30 -code=compute_30
 
 
# all
all: gpu cpu gpu2
 
cpu: fractaliseur.c
	$(CC) fractaliseur.c -o fraccpu $(CFLAGS)
 
gpu: fractaliseur.cu
	$(NVCC) fractaliseur.cu -o fracgpu $(NVFLAGS)

gpu2: fractaliseur2.cu
	$(NVCC) fractaliseur2.cu -o fracgpu2 $(NVFLAGS)
 
 
# clean
clean:
	rm -rf *.bak rm -rf *.o
 
# mrproper
mrproper: clean
	rm -rf fraccpu fracgpu
