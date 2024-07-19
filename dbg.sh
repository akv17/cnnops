#!/bin/bash
gcc main.c -g -fopenmp -O3 -lopenblas -lm && OMP_NUM_THREADS=16 gdb ./a.out
