#include <stdio.h>

/**
 * GPU kernel function: Compute one acceleration component at an index derived from current threadIdx and blockIdx on the GPU
 */
__global__ void gpu_insert(double *x0, double *y0, double *masses, double *ax, double *ay, int start_idx, int end_idx, int N)
{
	// Define gravitational constant (arb. unit, example value!)
	double G = 1e-3;

	// Define softening scale of Plummer profile (arb. unit, example value!)
	double epsilon_squared = pow(0.1, 2);

	// Get current index for which the acceleration is to be computed
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N)
	{
		// Get position
		double x_i = x0[i];
		double y_i = y0[i];

		double ax_sum = 0.;
		double ay_sum = 0.;

		// Directly sum the acceleration contributions from all particles without Fourier methods
		for (int j = start_idx; j < end_idx; ++j)
		{
			if (j == i)
				continue;

			// Get j-th particle position and mass
			double x_j = x0[j];
			double y_j = y0[j];
			double m_j = masses[j];

			// Intermediate quantities needed
			double dx = x_i - x_j;
			double dy = y_i - y_j;
			double dxsq = dx*dx;
			double dysq = dy*dy;
			double dr2 = dxsq + dysq + epsilon_squared;
			double dr = sqrt(dr2);
			double dr3 = dr*dr2;

			// Compute acceleration for the Plummer potential
			ax_sum += m_j * dx/dr3; 
			ay_sum += m_j * dy/dr3;
		}

		// Add contribution to sum
		ax[i] += -G*ax_sum;
		ay[i] += -G*ay_sum;

	}
}

/**
 * Auxiliary function
 */
int int_division_up(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

/**
 * Run the GPU computation, i.e. transfer data to GPU, call kernel, transfer data back
 */
void get_acceleration_vectors_GPU(double *x0, double *y0, double *masses, double *ax, double *ay, int N)
{
	// Define "device" quantities for coordinates, masses and accelerations
	double *dev_x0, *dev_y0, *dev_masses, *dev_ax, *dev_ay;
	int size = N *sizeof( double);
	cudaMalloc((void**)&dev_x0, size);
	cudaMalloc((void**)&dev_y0, size);
	cudaMalloc((void**)&dev_masses, size);
	cudaMalloc((void**)&dev_ax, size);
	cudaMalloc((void**)&dev_ay, size);

	// Transfer data from host (CPU) to device (GPU)
	cudaMemcpy(dev_x0, x0, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y0, y0, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_masses, masses, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ax, ax, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ay, ay, size, cudaMemcpyHostToDevice);

	// Set-up division for blocks and threads
	dim3 threads(N_t);
	dim3 blocks(int_division_up(N, N_t));

	// Run the kernel
	gpu_insert<<<blocks, threads>>>(dev_x0, dev_y0, dev_masses, dev_ax, dev_ay, 0, N, N);

	// Check for error
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	cudaDeviceSynchronize();

	// Transfer data back and free the GPU memory
	cudaMemcpy(ax, dev_ax, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(ay, dev_ay, size, cudaMemcpyDeviceToHost);
	cudaFree(dev_ax);
	cudaFree(dev_ay);
	cudaFree(dev_x0);
	cudaFree(dev_y0);
}
