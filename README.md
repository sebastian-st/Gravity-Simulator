

# Gravity simulator

Very simple N-body code to play around with gravity on a CUDA-capable GPU. The project is just for fun, visualising N-body problems and learning interesting stuff. The simulation uses Plummer potentials and is confined to a two-dimensional plane, which can be easily extended to 3D.

Demo video (opens a youtube link):

[![DEMO](https://yt-embed.herokuapp.com/embed?v=78bq11tZPqk)](https://www.youtube.com/watch?v=78bq11tZPqk "DEMO")

## Requirements 
- GPU with NVIDIA CUDA support
- Standard C++ library
- GSL if you want to use the random particle distribution generator

### Setup and usage

Edit the simulation settings and GPU specifics in the Makefile, run it and call the program without any arguments.
The code produces a txt output file for each simulation step or "snapshot" containing positions and velocities of all particles (format: "x y vx vy"). The data can be visualised with standard plotting tools provided by python or Gnuplot.
Have fun!

## Misc.

More detailed information might follow soon...

## Weblinks

http://sebastian.stapelberg.de

