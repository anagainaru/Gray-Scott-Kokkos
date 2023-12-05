/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * The solver is based on Hiroshi Watanabe's 2D Gray-Scott reaction diffusion
 * code available at: https://github.com/kaityo256/sevendayshpc/tree/master/day5
 */

#include "gray-scott.h"

#include <mpi.h>
#include <stdexcept> // runtime_error
#include <vector>

GSComm::GSComm(const Settings &settings, MPI_Comm comm)
: comm(comm), rand_pool(5374857)
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
    {
        size_x++;
    }
    if (py < settings.L % npy)
    {
        size_y++;
    }
    if (pz < settings.L % npz)
    {
        size_z++;
    }

    offset_x = (settings.L / npx * px) + std::min(settings.L % npx, px);
    offset_y = (settings.L / npy * py) + std::min(settings.L % npy, py);
    offset_z = (settings.L / npz * pz) + std::min(settings.L % npz, pz);

    MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
    MPI_Cart_shift(cart_comm, 1, 1, &down, &up);
    MPI_Cart_shift(cart_comm, 2, 1, &south, &north);
}

GSComm::~GSComm() {}

template<class MemSpace>
void GSComm::exchange_xy(
    Kokkos::View<double ***, MemSpace> localData) const
{
    // copy the first and last xy surface to a CPU structure 
    Kokkos::View<double **, MemSpace> firstXYGhost("firstGhost", size_x, size_y);
    Kokkos::View<double **, MemSpace> lastXYGhost("lastGhost", size_x, size_y);
    auto firstXYData = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kokkos::subview(localData, Kokkos::ALL, Kokkos::ALL, 1));
    auto lastXYData = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kokkos::subview(localData, Kokkos::ALL, Kokkos::ALL, size_z));
    MPI_Status st;

    // Send XY face z=size_z to north and receive z=0 from south
    MPI_Sendrecv(&lastXYData.data(), size_x * size_y, MPI_DOUBLE, north, 1,
                 &firstXYGhost.data(), size_x * size_y, MPI_DOUBLE, south, 1, cart_comm, &st);
    // Send XY face z=1 to south and receive z=size_z+1 from north
    MPI_Sendrecv(&firstXYData.data(), size_x * size_y, MPI_DOUBLE, south, 1,
                 &lastXYGhost.data(), size_x * size_y, MPI_DOUBLE, north, 1, cart_comm, &st);

    Kokkos::deep_copy(Kokkos::subview(localData, Kokkos::ALL, Kokkos::ALL, 0), firstXYGhost);
    Kokkos::deep_copy(Kokkos::subview(localData, Kokkos::ALL, Kokkos::ALL, size_z + 1), lastXYGhost);
}

template<class MemSpace>
void GSComm::exchange_xz(
    Kokkos::View<double ***, MemSpace> localData) const
{
    // copy the first and last xz surface to a CPU structure 
    Kokkos::View<double **, MemSpace> firstXZGhost("firstGhost", size_x, size_z);
    Kokkos::View<double **, MemSpace> lastXZGhost("lastGhost", size_x, size_z);
    auto firstXZData = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kokkos::subview(localData, Kokkos::ALL, 1, Kokkos::ALL));
    auto lastXZData = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kokkos::subview(localData, Kokkos::ALL, size_y, Kokkos::ALL));
    MPI_Status st;

    // Send XZ face y=size_y to up and receive y=0 from down
    MPI_Sendrecv(&lastXZData.data(), size_x * size_z, MPI_DOUBLE, up, 2,
                 &firstXZGhost.data(), size_x * size_z, MPI_DOUBLE, down, 2, cart_comm, &st);
    // Send XZ face y=1 to down and receive y=size_y+1 from up
    MPI_Sendrecv(&firstXZData.data(), size_x * size_z, MPI_DOUBLE, down, 2,
                 &lastXZGhost.data(), size_x * size_z, MPI_DOUBLE, up, 2, cart_comm, &st);

    Kokkos::deep_copy(Kokkos::subview(localData, Kokkos::ALL, 0, Kokkos::ALL), firstXZGhost);
    Kokkos::deep_copy(Kokkos::subview(localData, Kokkos::ALL, size_y + 1, Kokkos::ALL), lastXZGhost);
}

template<class MemSpace>
void GSComm::exchange_yz(Kokkos::View<double ***, MemSpace> localData) const
{
    // copy the first and last yz surface to a CPU structure 
    Kokkos::View<double **, MemSpace> firstYZGhost("firstGhost", size_y, size_z);
    Kokkos::View<double **, MemSpace> lastYZGhost("lastGhost", size_y, size_z);
    auto firstYZData = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kokkos::subview(localData, 1, Kokkos::ALL, Kokkos::ALL));
    auto lastYZData = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kokkos::subview(localData, size_x, Kokkos::ALL, Kokkos::ALL));
    MPI_Status st;

    // Send YZ face x=size_x to east and receive x=0 from west
    MPI_Sendrecv(&lastYZData.data(), size_y * size_z, MPI_DOUBLE, east, 3,
                 &firstYZGhost.data(), size_y * size_z, MPI_DOUBLE, west, 3, cart_comm, &st);
    // Send YZ face x=1 to west and receive x=size_x+1 from east
    MPI_Sendrecv(&firstYZData.data(), size_y * size_z, MPI_DOUBLE, west, 3,
                 &lastYZGhost.data(), size_y * size_z, MPI_DOUBLE, east, 3, cart_comm, &st);

    Kokkos::deep_copy(Kokkos::subview(localData, 0, Kokkos::ALL, Kokkos::ALL), firstYZGhost);
    Kokkos::deep_copy(Kokkos::subview(localData, size_x + 1, Kokkos::ALL, Kokkos::ALL), lastYZGhost);
}

template<class MemSpace>
void GSComm::ExchangeBorders(Kokkos::View<double ***, MemSpace> u,
                         Kokkos::View<double ***, MemSpace> v) const
{
    exchange_xy(u);
    exchange_xz(u);
    exchange_yz(u);

    exchange_xy(v);
    exchange_xz(v);
    exchange_yz(v);
}

template<class MemSpace>
void InitializeGSData(Kokkos::View<double ***, MemSpace> u, Kokkos::View<double **, MemSpace> v, Settings settings, GSComm simComm)
{
    Kokkos::deep_copy(u, 1.0);

    const int d = 6;
    auto const settingsL = static_cast<int>(settings.L);
    size_t const ox = simComm.offset_x, oy = simComm.offset_y, oz = simComm.offset_z;
    size_t const sx = simComm.size_x, sy = simComm.size_y;
    auto const min_z = std::max(settings.L / 2 - d, simComm.offset_z);
    auto const max_z = std::min(settings.L / 2 + d, simComm.offset_z + simComm.size_z);
    Kokkos::parallel_for(
        "init_buffers", Kokkos::RangePolicy<>(min_z, max_z), KOKKOS_LAMBDA(int z) {
            for (int y = settingsL / 2 - d; y < settingsL / 2 + d; y++)
            {
                if (y < static_cast<int>(oy))
                    continue;
                if (y >= static_cast<int>(oy + sy))
                    continue;
                for (int x = settingsL / 2 - d; x < settingsL / 2 + d; x++)
                {
                    if (x < static_cast<int>(ox))
                        continue;
                    if (x >= static_cast<int>(ox + sx))
                        continue;
                    u(x - ox + 1, y - oy + 1, z - oz + 1) = 0.25;
                    v(x - ox + 1, y - oy + 1, z - oz + 1) = 0.33;
                }
            }
        });
}

template<class MemSpace>
void IterateGS(Kokkos::View<double ***, MemSpace> u, Kokkos::View<double **, MemSpace> v, Settings settings, GSComm simComm)
{
    simComm.ExchangeBorders(u, v);
    ComputeNextIteration<MemSpace>(u, v, settings, simComm);
}

template<class MemSpace>
void ComputeNextIteration(Kokkos::View<double ***, MemSpace> u, Kokkos::View<double **, MemSpace> v, Settings settings, GSComm simComm)
{
    Kokkos::View<double ***, MemSpace> u2(simComm.size_x + 2, simComm.size_y + 2, simComm.size_z + 2);
    Kokkos::View<double ***, MemSpace> v2(simComm.size_x + 2, simComm.size_y + 2, simComm.size_z + 2);
    auto const Du = settings.Du;
    auto const Dv = settings.Dv;
    auto const dt = settings.dt;
    auto const F = settings.F;
    auto const k = settings.k;
    auto const noise = settings.noise;
    size_t const sx = simComm.size_x, sy = simComm.size_y, sz = simComm.size_z;
    auto const random_pool = simComm.rand_pool;
    Kokkos::parallel_for(
        "calc_gray_scott", Kokkos::RangePolicy<>(1, sz + 1), KOKKOS_LAMBDA(int z) {
            GSComm::RandomPool::generator_type generator = random_pool.get_state();
            double ts;
            for (int y = 1; y < static_cast<int>(sy) + 1; y++)
            {
                for (int x = 1; x < static_cast<int>(sx) + 1; x++)
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
                    du += noise * generator.frand(-1.f, 1.f);
                    u2(x, y, z) = u(x, y, z) + du * dt;
                    v2(x, y, z) = v(x, y, z) + dv * dt;
                }
            }
            random_pool.free_state(generator);
        });
    std::swap(u, u2);
    std::swap(v, v2);
}