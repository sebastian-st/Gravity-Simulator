#include <stdio.h>
#include <array> // for std::array
#include <vector> // for std::vector

using namespace std;

#include "particles.h"
#include "add.h"

// Query particle coordinates
array<double, 2> particle::get_pos()
{
	array<double, 2> coord = {x,y};
	return coord;
}

// Query particle velocity components
array<double, 2> particle::get_velocity()
{
	array<double, 2> velocity = {vx,vy};
	return velocity;
}

// Query particle mass
double particle::get_mass() 
{ 
	return mass; 
}

// Shift the particle to a new position (x_,y_)
void particle::move(double x_, double y_)
{
	x = x_;
	y = y_;
}

// Give the particle an acceleration (ax,ay)
void particle::accelerate(double ax, double ay, double time_step_size)
{
	vx += ax * time_step_size;
	vy += ay * time_step_size;
	double new_x = x + vx * time_step_size;
	double new_y = y + vy * time_step_size;
	move(new_x, new_y);
}



// Query coordinates and mass
void ensemble::get_particle_data(double *x0, double *y0, double *m)
{
	int N = members.size();
	for (int i = 0; i < N; ++i)
	{
		x0[i] = members[i].x;
		y0[i] = members[i].y;
		m[i] = members[i].mass;
	}
}

// Query velocity components as a vector
vector<array<double, 2>> ensemble::get_velocities()
{
	int N = members.size();
	vector<array<double, 2>> velocities(N);
	for (int i = 0; i < N; ++i)
		velocities[i] = members[i].get_velocity();
	return velocities;
}

// Propagate each particle one time-step forward, computing the accelerations on the GPU
void ensemble::propagate(double delta_t)
{
	// Define "host" quantities (CPU)
	double *host_x0, *host_y0, *host_masses, *host_ax, *host_ay;
	int N = members.size();
	int mem_size = sizeof(double) * N;
	host_ax = (double *)malloc(mem_size);
	host_ay = (double *)malloc(mem_size);
	host_x0 = (double *)malloc(mem_size);
	host_y0 = (double *)malloc(mem_size);
	host_masses = (double *)malloc(mem_size);

	// Fill the host arrays with the data
	get_particle_data(host_x0, host_y0, host_masses);

	// Initialise acceleration arrays with zero
	for (int i = 0; i < N; ++i)
	{
		host_ax[i] = 0.;
		host_ay[i] = 0.;
	}

	// Copy data to GPU ("device"), compute accelerations there and transfer the results back to CPU
	get_acceleration_vectors_GPU(host_x0, host_y0, host_masses, host_ax, host_ay, N);

	// Apply the accelerations to each particle
	for (int i = 0; i < N; ++i)
		members[i].accelerate(host_ax[i], host_ay[i], delta_t);

	// Free host arrays
	free(host_x0);
	free(host_y0);
	free(host_masses);
	free(host_ax);
	free(host_ay);
}


