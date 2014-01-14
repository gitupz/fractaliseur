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
all: gpu cpu
 
cpu: fractaliseur.c
	$(CC) fractaliseur.c -o fraccpu $(CFLAGS)
 
gpu: fractaliseur.cu
	$(NVCC) fractaliseur.cu -o fracgpu $(NVFLAGS)
 
 
# clean
clean:
	rm -rf *.bak rm -rf *.o
 
# mrproper
mrproper: clean
	rm -rf fraccpu fracgpu
