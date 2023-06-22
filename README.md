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

## Configurations

Flower-pattern: F=0.055 ; k=0.062

Mazes-pattern: F =0.029 ; k=0.057

Mitosis-pattern: F=0.028 ; k=0.062

Solitons-pattern: F=0.03 ; k=0.06
