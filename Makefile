# SIMULATION SETTINGS (EXAMPLE VALUES! ARB. UNITS...):
OUTFILE_PREFIX = "example_folder/data_"# where we assume the filename format: data_{step}.txt
N_PARTICLES = 100
N_TIMESTEPS = 5000
TIMESTEP_SIZE = 0.03
RESUME_FROM_PREVIOUS_FILE = 0
PREVIOUS_FILE_IDX = 0

# CUSTOM DEVICE-DEPENDENT SETTINGS:
GPU_VIRT_ARCH = sm_30
MACHINE_ARCH = 64
THREADS_PER_BLOCK = 1024

# STANDARD OPTIONS
MAIN_TARGET_NAME = gravity_simulator
CXX := g++
CXXFLAGS := -Wall -Wpedantic -std=c++11 -flto 
LDFLAGS := -lgsl -lstdc++ 
NVCCFLAGS := --std c++11 -arch=$(GPU_VIRT_ARCH) --machine $(MACHINE_ARCH) -use_fast_math
NVCCFLAGS_END := --shared -lcuda --compiler-options '-fPIC'
CUSTOM_OPTS = -DOUTFILE_PREFIX=\"$(OUTFILE_PREFIX)\" -DN_PARTICLES=$(N_PARTICLES) -DN_TIMESTEPS=$(N_TIMESTEPS) -DTIMESTEP_SIZE=$(TIMESTEP_SIZE) -DRESUME_FROM_PREVIOUS_FILE=$(RESUME_FROM_PREVIOUS_FILE) -DPREVIOUS_FILE_IDX=$(PREVIOUS_FILE_IDX)

all: gpu_add main

gpu_add:
	nvcc $(NVCCFLAGS) -o shared/add.so -DN_t=$(THREADS_PER_BLOCK) src/add.cu $(NVCCFLAGS_END)

main:
	$(CXX) $(CXXFLAGS) -o $(MAIN_TARGET_NAME) src/main.cpp src/particles.cpp shared/add.so $(LDFLAGS) -Wl,-R,'$$ORIGIN' $(CUSTOM_OPTS)

.PHONY: all clean

clean:
	rm src/add.so && rm $(MAIN_TARGET_NAME)
