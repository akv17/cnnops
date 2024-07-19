#ifndef TIMER_H
#define TIMER_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

struct timeval tval_before, tval_after, tval_result;


void timer_start() {
    gettimeofday(&tval_before, NULL);
}


void timer_end() {
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
}


void timer_print() {
    printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
}


void timer_end_and_print() {
    timer_end();
    timer_print();
}


#endif
