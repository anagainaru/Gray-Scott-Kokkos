/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 */

#ifndef __RESTART_H__
#define __RESTART_H__

#include "gray-scott.h"

#include <adios2.h>
#include <mpi.h>

template<class MemSpace>
void WriteCkpt(MPI_Comm comm, const int step, const Settings &settings, const GSComm &sim,
               adios2::IO io, const Kokkos::View<double ***, MemSpace> &u,
               const Kokkos::View<double ***, MemSpace> &v);
template<class MemSpace>
int ReadRestart(MPI_Comm comm, const Settings &settings, GSComm &sim, adios2::IO io,
                const Kokkos::View<double ***, MemSpace> &u, const Kokkos::View<double ***, MemSpace> &v);

#endif
