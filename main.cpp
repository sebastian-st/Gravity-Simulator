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

/**
 * Class defining a particle and its physical properties
 */
class particle
{
	private:
		// Particle coordinates
		double x = 0., y = 0.;

		// Particle velocities
		double vx = 0., vy = 0.;

		// Particle mass
		double mass = 0.;

	public:
		friend class ensemble;

		// Constructors
		particle() {};
		particle(double x_, double y_, double vx_, double vy_, double mass_) : x(x_), y(y_), vx(vx_), vy(vy_), mass(mass_) {};

		// Query particle coordinates
		array<double, 2> get_pos()
		{
			array<double, 2> coord = {x,y};
			return coord;
		};

		// Query particle velocity components
		array<double, 2> get_velocity()
		{
			array<double, 2> velocity = {vx,vy};
			return velocity;
		};

		// Query particle mass
		double get_mass() { return mass; }

		// Shift the particle to a new position (x_,y_)
		void move(double x_, double y_)
		{
			x = x_;
			y = y_;
		};

		// Give the particle an acceleration (ax,ay)
		void accelerate(double ax, double ay, double time_step_size)
		{
			vx += ax * time_step_size;
			vy += ay * time_step_size;
			double new_x = x + vx * time_step_size;
			double new_y = y + vy * time_step_size;
			move(new_x, new_y);
		};
};

/**
 * Class defining an ensemble of particles and their statistical properties
 */
class ensemble
{
	private:
		vector<particle> members = {};
	public:
		// Constructor
		ensemble(vector<particle> &members_) : members(members_) {};

		// Query coordinates and mass
		void get_particle_data(double *x0, double *y0, double *m)
		{
			int N = members.size();
			for (int i = 0; i < N; ++i)
			{
				x0[i] = members[i].x;
				y0[i] = members[i].y;
				m[i] = members[i].mass;
			}
		};

		// Query velocity components as a vector
		vector<array<double, 2>> get_velocities()
		{
			int N = members.size();
			vector<array<double, 2>> velocities(N);
			for (int i = 0; i < N; ++i)
				velocities[i] = members[i].get_velocity();
			return velocities;
		};

		// Propagate each particle one time-step forward, computing the accelerations on the GPU
		void propagate(double delta_t)
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
		};
};

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
	cout << N_particles << " " << N_timesteps << " " << timestep_size << " " << outfile_prefix << " " << resume_from_file << " " << initial_step_idx << endl;

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
