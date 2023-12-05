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
    private:
    int rank, procs;

    int west, east, up, down, north, south;
    MPI_Comm comm;
    MPI_Comm cart_comm;

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

    GSComm(const Settings &settings, MPI_Comm comm): comm(comm), rand_pool(5374857)
	{
		int dims[3] = {};
		const int periods[3] = {1, 1, 1};
		int coords[3] = {};

		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &procs);

		MPI_Dims_create(procs, 3, dims);
		npx = dims[0];
		npy = dims[1];
		npz = dims[2];

		MPI_Cart_create(comm, 3, dims, periods, 0, &cart_comm);
		MPI_Cart_coords(cart_comm, rank, 3, coords);
		px = coords[0];
		py = coords[1];
		pz = coords[2];

		size_x = settings.L / npx;
		size_y = settings.L / npy;
		size_z = settings.L / npz;

		if (px < settings.L % npx)
			size_x++;
		if (py < settings.L % npy)
			size_y++;
		if (pz < settings.L % npz)
			size_z++;

		offset_x = (settings.L / npx * px) + std::min(settings.L % npx, px);
		offset_y = (settings.L / npy * py) + std::min(settings.L % npy, py);
		offset_z = (settings.L / npz * pz) + std::min(settings.L % npz, pz);

		MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
		MPI_Cart_shift(cart_comm, 1, 1, &down, &up);
		MPI_Cart_shift(cart_comm, 2, 1, &south, &north);
	};

    ~GSComm(){};

    // Exchange XY faces with north/south
    template<class MemSpace>
    void exchange_xy(Kokkos::View<double ***, MemSpace> localData) const
	{
        int sx = size_x + 2;
        int sy = size_y + 2;
		// copy the first and last xy surface to a CPU structure 
		Kokkos::View<double **, MemSpace> firstXYGhost("firstGhostPlane", sx, sy);
		Kokkos::View<double **, MemSpace> lastXYGhost("lastGhostPlane", sx, sy);
        Kokkos::View<double **, Kokkos::HostSpace> firstXYData("firstDataPlane", sx, sy);
        Kokkos::deep_copy(firstXYData, Kokkos::subview(localData, Kokkos::ALL, Kokkos::ALL, 1));
        Kokkos::View<double **, Kokkos::HostSpace> lastXYData("lastDataPlane", sx, sy);
        Kokkos::deep_copy(lastXYData, Kokkos::subview(localData, Kokkos::ALL, Kokkos::ALL, size_z));
		MPI_Status st;

		// Send XY face z=size_z to north and receive z=0 from south
		MPI_Sendrecv(lastXYData.data(), sx * sy, MPI_DOUBLE, north, 1,
					 firstXYGhost.data(), sx * sy, MPI_DOUBLE, south, 1, cart_comm, &st);
		// Send XY face z=1 to south and receive z=size_z+1 from north
		MPI_Sendrecv(firstXYData.data(), sx * sy, MPI_DOUBLE, south, 1,
					 lastXYGhost.data(), sx * sy, MPI_DOUBLE, north, 1, cart_comm, &st);

		Kokkos::deep_copy(Kokkos::subview(localData, Kokkos::ALL, Kokkos::ALL, 0), firstXYGhost);
		Kokkos::deep_copy(Kokkos::subview(localData, Kokkos::ALL, Kokkos::ALL, size_z + 1), lastXYGhost);
	};

    // Exchange XZ faces with up/down
    template<class MemSpace>
    void exchange_xz(Kokkos::View<double ***, MemSpace> localData) const
	{
        int sx = size_x + 2;
        int sz = size_z + 2;
		// copy the first and last xz surface to a CPU structure 
		Kokkos::View<double **, MemSpace> firstXZGhost("firstGhostPlane", sx, sz);
		Kokkos::View<double **, MemSpace> lastXZGhost("lastGhostPlane", sx, sz);
        Kokkos::View<double **, Kokkos::HostSpace> firstXZData("firstDataPlane", sx, sz);
        Kokkos::deep_copy(firstXZData, Kokkos::subview(localData, Kokkos::ALL, 1, Kokkos::ALL));
        Kokkos::View<double **, Kokkos::HostSpace> lastXZData("lastDataPlane", sx, sz);
        Kokkos::deep_copy(lastXZData, Kokkos::subview(localData, Kokkos::ALL, size_y, Kokkos::ALL));
		MPI_Status st;

		// Send XZ face y=size_y to up and receive y=0 from down
		MPI_Sendrecv(lastXZData.data(), sx * sz, MPI_DOUBLE, up, 2,
					 firstXZGhost.data(), sx * sz, MPI_DOUBLE, down, 2, cart_comm, &st);
		// Send XZ face y=1 to down and receive y=size_y+1 from up
		MPI_Sendrecv(firstXZData.data(), sx * sz, MPI_DOUBLE, down, 2,
					 lastXZGhost.data(), sx * sz, MPI_DOUBLE, up, 2, cart_comm, &st);

		Kokkos::deep_copy(Kokkos::subview(localData, Kokkos::ALL, 0, Kokkos::ALL), firstXZGhost);
		Kokkos::deep_copy(Kokkos::subview(localData, Kokkos::ALL, size_y + 1, Kokkos::ALL), lastXZGhost);
	};

    // Exchange YZ faces with west/east
    template<class MemSpace>
    void exchange_yz(Kokkos::View<double ***, MemSpace> localData) const
	{
        int sy = size_y + 2;
        int sz = size_z + 2;
		// copy the first and last yz surface to a CPU structure 
		Kokkos::View<double **, MemSpace> firstYZGhost("firstGhostPlane", sy, sz);
		Kokkos::View<double **, MemSpace> lastYZGhost("lastGhostPlane", sy, sz);
        Kokkos::View<double **, Kokkos::HostSpace> firstYZData("firstDataPlane", sy, sz);
        Kokkos::deep_copy(firstYZData, Kokkos::subview(localData, 1, Kokkos::ALL, Kokkos::ALL));
        Kokkos::View<double **, Kokkos::HostSpace> lastYZData("lastDataPlane", sy, sz);
        Kokkos::deep_copy(lastYZData, Kokkos::subview(localData, size_x, Kokkos::ALL, Kokkos::ALL));
		MPI_Status st;

		// Send YZ face x=size_x to east and receive x=0 from west
		MPI_Sendrecv(lastYZData.data(), sy * sz, MPI_DOUBLE, east, 3,
					 firstYZGhost.data(), sy * sz, MPI_DOUBLE, west, 3, cart_comm, &st);
		// Send YZ face x=1 to west and receive x=size_x+1 from east
		MPI_Sendrecv(firstYZData.data(), sy * sz, MPI_DOUBLE, west, 3,
					 lastYZGhost.data(), sy * sz, MPI_DOUBLE, east, 3, cart_comm, &st);

		Kokkos::deep_copy(Kokkos::subview(localData, 0, Kokkos::ALL, Kokkos::ALL), firstYZGhost);
		Kokkos::deep_copy(Kokkos::subview(localData, size_x + 1, Kokkos::ALL, Kokkos::ALL), lastYZGhost);
	};

    // Exchange faces with neighbors
    template<class MemSpace>
    void ExchangeBorders(Kokkos::View<double ***, MemSpace> u,
                  Kokkos::View<double ***, MemSpace> v) const
	{
		exchange_xy(u);
		exchange_xz(u);
		exchange_yz(u);

		exchange_xy(v);
		exchange_xz(v);
		exchange_yz(v);
	};

};

template<class MemSpace>
void InitializeGSData(const Kokkos::View<double ***, MemSpace> &u, const Kokkos::View<double ***, MemSpace> &v, Settings settings, GSComm simComm)
{
    const int d = 6;
    auto const settingsL = static_cast<int>(settings.L);
    size_t const ox = simComm.offset_x, oy = simComm.offset_y, oz = simComm.offset_z;
    size_t const sx = simComm.size_x, sy = simComm.size_y, sz = simComm.size_z;
    auto const min_x = std::max(settings.L / 2 - d, simComm.offset_x);
    auto const max_x = std::min(settings.L / 2 + d, simComm.offset_x + simComm.size_x);
    Kokkos::parallel_for(
        "init_buffers", Kokkos::RangePolicy<>(min_x, max_x), KOKKOS_LAMBDA(int x) {
            for (int y = settingsL / 2 - d; y < settingsL / 2 + d; y++)
            {
                if (y < static_cast<int>(oy))
                    continue;
                if (y >= static_cast<int>(oy + sy))
                    continue;
                for (int z = settingsL / 2 - d; z < settingsL / 2 + d; z++)
                {
                    if (z < static_cast<int>(oz))
                        continue;
                    if (z >= static_cast<int>(oz + sz))
                        continue;
                    u(x - ox, y - oy, z - oz) = 0.25;
                    v(x - ox, y - oy, z - oz) = 0.33;
                }
            }
        });
};

template<class MemSpace>
void ComputeNextIteration(Kokkos::View<double ***, MemSpace> &u, Kokkos::View<double ***, MemSpace> &v, Kokkos::View<double ***, MemSpace> &u2, Kokkos::View<double ***, MemSpace> &v2, Settings settings, GSComm simComm)
{
    auto const Du = settings.Du;
    auto const Dv = settings.Dv;
    auto const dt = settings.dt;
    auto const F = settings.F;
    auto const k = settings.k;
    auto const noise = settings.noise;
    size_t const sx = simComm.size_x, sy = simComm.size_y, sz = simComm.size_z;
    auto const random_pool = simComm.rand_pool;
    Kokkos::parallel_for(
        "calc_gray_scott", Kokkos::RangePolicy<>(1, sx + 1), KOKKOS_LAMBDA(int x) {
            GSComm::RandomPool::generator_type generator = random_pool.get_state();
            double ts;
            for (int y = 1; y < static_cast<int>(sy) + 1; y++)
            {
                for (int z = 1; z < static_cast<int>(sz) + 1; z++)
                {
                    double du, dv;
                    // laplacian for u
                    ts = 0;
                    ts += u(x - 1, y, z);
                    ts += u(x + 1, y, z);
                    ts += u(x, y - 1, z);
                    ts += u(x, y + 1, z);
                    ts += u(x, y, z - 1);
                    ts += u(x, y, z + 1);
                    ts += -6.0 * u(x, y, z);
                    ts /= 6.0;
                    du = Du * ts;

                    // laplacian for v
                    ts = 0;
                    ts += v(x - 1, y, z);
                    ts += v(x + 1, y, z);
                    ts += v(x, y - 1, z);
                    ts += v(x, y + 1, z);
                    ts += v(x, y, z - 1);
                    ts += v(x, y, z + 1);
                    ts += -6.0 * v(x, y, z);
                    ts /= 6.0;
                    dv = Dv * ts;

                    du += (-u(x, y, z) * v(x, y, z) * v(x, y, z) +
                           F * (1.0 - u(x, y, z)));
                    dv += (u(x, y, z) * v(x, y, z) * v(x, y, z) -
                           (F + k) * v(x, y, z));
                    //du += noise * generator.frand(-1.f, 1.f);
                    u2(x, y, z) = u(x, y, z) + du * dt;
                    v2(x, y, z) = v(x, y, z) + dv * dt;
                }
            }
            random_pool.free_state(generator);
        });
    std::swap(u, u2);
    std::swap(v, v2);
};

template<class MemSpace>
void IterateGS(Kokkos::View<double ***, MemSpace> &u, Kokkos::View<double ***, MemSpace> &v, Kokkos::View<double ***, MemSpace> &u2, Kokkos::View<double ***, MemSpace> &v2, Settings settings, GSComm simComm)
{
    simComm.ExchangeBorders(u, v);
    ComputeNextIteration<MemSpace>(u, v, u2, v2, settings, simComm);
};

#endif
