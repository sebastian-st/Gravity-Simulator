#ifndef PARTICLES_H
#define PARTICLES_H

#include <array> // for std::array
#include <vector> // for std::vector

using namespace std;

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
		array<double, 2> get_pos();
	
		// Query particle velocity components
		array<double, 2> get_velocity();

		// Query particle mass
		double get_mass();

		// Shift the particle to a new position (x_,y_)
		void move(double x_, double y_);

		// Give the particle an acceleration (ax,ay)
		void accelerate(double ax, double ay, double time_step_size);
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
		void get_particle_data(double *x0, double *y0, double *m);

		// Query velocity components as a vector
		vector<array<double, 2>> get_velocities();

		// Propagate each particle one time-step forward, computing the accelerations on the GPU
		void propagate(double delta_t);
};

#endif
