/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 */

#ifndef __WRITER_H__
#define __WRITER_H__

#include <adios2.h>
#include <mpi.h>
#include <utility>

#include "gray-scott.h"

class Writer
{
public:
    Writer(const Settings &settings, const GSComm &sim, adios2::IO io);
    void open(const std::string &fname, bool append);
    void close();

    template <class MemSpace>
    void write(int step, const GSComm &sim, const Kokkos::View<double ***, MemSpace> &u,
               const Kokkos::View<double ***, MemSpace> &v)
    {
        if (!sim.size_x || !sim.size_y || !sim.size_z)
        {
            writer.BeginStep();
            writer.EndStep();
            return;
        }

        if (settings.adios_memory_selection)
        {
            writer.BeginStep();
            writer.Put<int>(var_step, &step);
            writer.Put<double>(var_u, u.data());
            writer.Put<double>(var_v, v.data());
            writer.EndStep();
        }
        else if (settings.adios_span)
        {
            std::cout << "ADIOS2 Span with Kokkos currently not supported" << std::endl;
        }
        else
        {
            int dx = sim.size_x + 1;
            int dy = sim.size_y + 1;
            int dz = sim.size_z + 1;
            Kokkos::View<double ***, MemSpace> u_noghost("NoGhostU", sim.size_x, sim.size_y,
                                                         sim.size_z);
            Kokkos::deep_copy(u_noghost,
                              Kokkos::subview(u, std::make_pair(1, dx), std::make_pair(1, dy),
                                              std::make_pair(1, dz)));
            Kokkos::View<double ***, MemSpace> v_noghost("NoGhostV", sim.size_x, sim.size_y,
                                                         sim.size_z);
            Kokkos::deep_copy(v_noghost,
                              Kokkos::subview(v, std::make_pair(1, dx), std::make_pair(1, dy),
                                              std::make_pair(1, dz)));

            writer.BeginStep();
            writer.Put<int>(var_step, &step);
            writer.Put<double>(var_u, u_noghost.data());
            writer.Put<double>(var_v, v_noghost.data());
            writer.EndStep();
        }
    };

protected:
    Settings settings;

    adios2::IO io;
    adios2::Engine writer;
    adios2::Variable<double> var_u;
    adios2::Variable<double> var_v;
    adios2::Variable<int> var_step;
};

#endif
