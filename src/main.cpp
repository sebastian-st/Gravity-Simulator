#include <stdio.h>
#include <iostream> // for std::cout()
#include <fstream> // for ofstream
#include <array> // for std::array
#include <vector> // for std::vector
#include <string>

#include <random>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

using namespace std;

#include "add.h"
#include "particles.h"

/**
 * Main routine: Create particle ensemble, configure and run simulation
 */
int main()
{
	// Settings (arbitrary units)
	int N_particles = N_PARTICLES;
	int N_timesteps = N_TIMESTEPS;
	double timestep_size = TIMESTEP_SIZE;
	string outfile_prefix = OUTFILE_PREFIX;
	bool resume_from_file = RESUME_FROM_PREVIOUS_FILE;
	int initial_step_idx = PREVIOUS_FILE_IDX;

	// Define random number generator and particle vector
	gsl_rng *rn_gen = gsl_rng_alloc(gsl_rng_ranlux389);
	vector<particle> particles(N_particles);

	// If wished, get initial conditions from the previous state of an existing file
	if (resume_from_file)
	{
		ifstream fh;
		fh.open(outfile_prefix+to_string(initial_step_idx)+".txt");
		for (int c = 0; c < N_particles; ++c) // NOTE: N_particles needs to agree with the file line count!
		{
			double x, y, vx, vy;
			fh >> x >> y >> vx >> vy;
			particle p(x, y, vx, vy, 0.5);
			particles[c] = p;
		}
		fh.close();
	}
	// Otherwise, create new initial conditions
	else
	{
		for (int c = 0; c < N_particles; ++c)
		{
			// ---- Define your own initial conditions here...! ---

			// Example: Populate the position and velocity space with a Gaussian particle distribution around (0,0) in both cases
			double pos_spread = 1.;
			double velocity_spread = 0.1;
			double x = gsl_ran_gaussian(rn_gen, pos_spread);
			double y = gsl_ran_gaussian(rn_gen, pos_spread);
			double vx0 = gsl_ran_gaussian(rn_gen, velocity_spread);
			double vy0 = gsl_ran_gaussian(rn_gen, velocity_spread);
			particle p(x, y, vx0, vy0, 100.);
			particles[c] = p;
		}
	}

	cout << "Creating ensemble of " << N_particles << " particles" << endl;
	ensemble e(particles);

	//
	int mem_size = sizeof(double) * N_particles;
	double *x0 = (double *)malloc(mem_size);
	double *y0 = (double *)malloc(mem_size);
	double *masses = (double *)malloc(mem_size);

	// Loop over the timesteps, propagate the particles forward and write output to file
	int N_total_steps = initial_step_idx+N_timesteps;
	for (int t = initial_step_idx; t < N_total_steps; ++t)
	{
		ofstream fh;
		fh.open(outfile_prefix+to_string(t)+".txt", ios::trunc);
		if (!fh.good())
		{
			cout << "Error opening output file for writing" << endl;
			exit(1);
		}
		fh.close();

		cout << "\r" << "Processing step " << t << " (" << (int)(100*t/N_total_steps) << "%)" << flush;
		e.get_particle_data(x0, y0, masses);
		vector<array<double, 2>> v = e.get_velocities();

		for (int i = 0; i < N_particles; ++i)
		{
			fh.open(outfile_prefix+to_string(t)+".txt", ios::app);
			fh << x0[i] << " " << y0[i] << " " << v[i][0] << " " << v[i][1] << endl;
			fh.close();
		}

		e.propagate(timestep_size);
	}
	cout << endl;
}
