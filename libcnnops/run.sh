#!/bin/bash
gcc -fopenmp -O3 main.c && OMP_NUM_THREADS=16 ./a.out
