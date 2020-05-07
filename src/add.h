#ifndef ADD_H
#define ADD_H

/**
 * Compute acceleration vectors on GPU
 */
void get_acceleration_vectors_GPU(double *x0, double *y0, double *masses, double *ax, double *ay, int N);

#endif
