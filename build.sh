#!/bin/bash
cd libmatmul && gcc matmul.c -O3 -fPIC -fopenmp -shared -o libmatmul.so