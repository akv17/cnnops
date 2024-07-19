#!/bin/bash
gcc main.c -O3 -lblas -lm && OMP_NUM_THREADS=16 ./a.out
