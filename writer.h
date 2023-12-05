/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 */

#ifndef __WRITER_H__
#define __WRITER_H__

#include <adios2.h>
#include <mpi.h>

#include "gray-scott.h"

class Writer
{
public:
    Writer(const Settings &settings, const GSComm &sim, adios2::IO io);
    void open(const std::string &fname, bool append);
    template <class MemSpace>
    void write(int step, const GSComm &sim, const Kokkos::View<double ***, MemSpace> &u,
                       const Kokkos::View<double ***, MemSpace> &v);
    void close();

protected:
    Settings settings;

    adios2::IO io;
    adios2::Engine writer;
    adios2::Variable<double> var_u;
    adios2::Variable<double> var_v;
    adios2::Variable<int> var_step;
};

#endif