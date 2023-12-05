/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 */

#ifndef __GRAY_SCOTT_H__
#define __GRAY_SCOTT_H__

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <mpi.h>

#include "settings.h"

class GSComm
{
    public:
    // Dimension of process grid
    size_t npx, npy, npz;
    // Coordinate of this rank in process grid
    size_t px, py, pz;
    // Dimension of local array
    size_t size_x, size_y, size_z;
    // Offset of local array in the global array
    size_t offset_x, offset_y, offset_z;

    using RandomPool = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;
    RandomPool rand_pool;

    GSComm(const Settings &settings, MPI_Comm comm);
    ~GSComm();

    // Exchange faces with neighbors
    template<class MemSpace>
    void ExchangeBorders(Kokkos::View<double ***, MemSpace> u,
                  Kokkos::View<double ***, MemSpace> v) const;
    // Exchange XY faces with north/south
    template<class MemSpace>
    void exchange_xy(Kokkos::View<double ***, MemSpace> local_data) const;
    // Exchange XZ faces with up/down
    template<class MemSpace>
    void exchange_xz(Kokkos::View<double ***, MemSpace> local_data) const;
    // Exchange YZ faces with west/east
    template<class MemSpace>
    void exchange_yz(Kokkos::View<double ***, MemSpace> local_data) const;

    private:
    int rank, procs;

    int west, east, up, down, north, south;
    MPI_Comm comm;
    MPI_Comm cart_comm;
};

template<class MemSpace>
void InitializeGSData(Kokkos::View<double ***, MemSpace> u, Kokkos::View<double ***, MemSpace> v, Settings settings, GSComm simComm);

template<class MemSpace>
void IterateGS(Kokkos::View<double ***, MemSpace> u, Kokkos::View<double ***, MemSpace> v, Settings settings, GSComm simComm);

template<class MemSpace>
void ComputeNextIteration(Kokkos::View<double ***, MemSpace> u, Kokkos::View<double **, MemSpace> v, Settings settings, GSComm simComm);
#endif
