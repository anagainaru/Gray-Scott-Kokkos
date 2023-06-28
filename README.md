# Gray-Scott using Kokkos and MPI

The Gray-Scott reaction diffusion model simulates a chemical reaction between two substances `U` and `V`, both of which diffuse over time. During the reaction `U` gets used up, while `V` is produced. The simulation is writing the `U` and `V` data every x step using POSIX, ADIOS2 or HDF5.

The Gray-Scott system is defined by two equations that describe the behaviour of two reacting substances (from [here](https://itp.uni-frankfurt.de/~gros/StudentProjects/Projects_2020/projekt_schulz_kaefer)):
```
u_t = - u * v^2 + F * (1 - u) + Du *  Δu + noise * randn(-1,1)
v_t = u * v^2 - (F + k) * v + Dv * Δv
```

`F` describes the rate at which `U` is replenished externally and `k` the rate at which `V` is removed at every step. Different values for the two rate will lead to different patterns.

The code in this repo is a stand alone code used in the [ADIOS2 example](https://github.com/ornladios/ADIOS2-Examples/tree/master/source/gpu/gray-scott-kokkos) allowing different I/O strategies and initial configurations.

## Running the code

The code requires Kokkos and MPI. Optionally adios2 can be used for writing and reading the output of the simulation.

To compile the code without adios2 (using POSIX for writing/reading):
```
 cmake -DKokkos_ROOT=path/to/Kokkos/install -D CMAKE_CXX_STANDARD=17 -D CMAKE_CXX_EXTENSIONS=OFF -DCMAKE_CXX_COMPILER=${CXX_compiler}  -DCMAKE_C_COMPILER=gcc ..
```
To link to adios2 the parth to the install path must be provided:
```
-Dadios2_ROOT=/path/to/adios/install
```
The CXX compiler needs to point to the nvcc_wrapper when running on CUDA architecture:
`/path/to/kokkos/bin/nvcc_wrapper`.

### Running the plot scripts

Running the code:
```
mpirun -n 2 ./bin/adios2-gray-scott-kokkos ../gs-settings.json

```
Running the plot function:
```
python ../source/cpp/gray-scott/plot/gsplot.py gs.bp/
```

When adios2 is used for I/O the plot and simulation can run in parallel and stream data between each other.
```
mpirun -n 2 ./bin/adios2-gray-scott-kokkos ../gs-settings.json & python ../source/cpp/gray-scott/plot/gsplot.py gs.bp/
```

## Configurations

Flower-pattern: F=0.055 ; k=0.062

Mazes-pattern: F =0.029 ; k=0.057

Mitosis-pattern: F=0.028 ; k=0.062

Solitons-pattern: F=0.03 ; k=0.06
