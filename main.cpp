#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

//#include <adios2.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "settings.hpp"
#include "simulation.hpp"
//#include "timer.hpp"
#include "writer.hpp"

grayscott::Simulation3D create_simulation(grayscott::Settings settings,
                                          int rank, int procs,
                                          const MPI_Comm &comm)
{
    // Dimension of process grid
    size_t npx, npy, npz;
    // Coordinate of this rank in process grid
    size_t px, py, pz;

    MPI_Comm cart_comm;
    int dims[3] = {};
    MPI_Dims_create(procs, 3, dims);
    npx = dims[0];
    npy = dims[1];
    npz = dims[2];

    const int periods[3] = {1, 1, 1};
    MPI_Cart_create(comm, 3, dims, periods, 0, &cart_comm);

    int coords[3] = {};
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    px = coords[0];
    py = coords[1];
    pz = coords[2];

    std::vector<size_t> size(3);
    size[0] = settings.L / npx;
    size[1] = settings.L / npy;
    size[2] = settings.L / npz;

    if (px < settings.L % npx)
    {
        size[0]++;
    }
    if (py < settings.L % npy)
    {
        size[1]++;
    }
    if (pz < settings.L % npz)
    {
        size[2]++;
    }

    std::vector<size_t> offset(3);
    offset[0] = (settings.L / npx * px) + std::min(settings.L % npx, px);
    offset[1] = (settings.L / npy * py) + std::min(settings.L % npy, py);
    offset[2] = (settings.L / npz * pz) + std::min(settings.L % npz, pz);
    return grayscott::Simulation3D(settings, cart_comm, size, offset);
}

int main(int argc, char **argv)
{
    int provided;
    int rank, procs, wrank;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

    // const unsigned int color = 1;
    const MPI_Comm comm = MPI_COMM_WORLD;
    // MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &comm);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    if (argc < 2)
    {
        if (rank == 0)
        {
            std::cerr << "Too few arguments" << std::endl;
            std::cerr << "Usage: gray-scott settings.json" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    Kokkos::initialize(argc, argv);
    {
        const grayscott::Settings settings(argv[1]);
        grayscott::Simulation3D sim =
            create_simulation(settings, rank, procs, comm);
        grayscott::Writer(settings, sim);
        /*
                Writer writer;
                adios2::ADIOS adios(settings.adios_config, comm);
                adios2::IO io_main = adios.DeclareIO("SimulationOutput");

                Writer writer_main(settings, sim, io_main);
                writer_main.open(settings.output, (restart_step > 0));

        */
        if (rank == 0)
        {
            //            writer.print_settings();
            std::cout << "========================================"
                      << std::endl;
            settings.print();
            std::cout << "========================================"
                      << std::endl;
        }

        /*
        #ifdef ENABLE_TIMERS
                Timer timer_total;
                Timer timer_compute;
                Timer timer_write;

                std::ostringstream log_fname;
                log_fname << "gray_scott_pe_" << rank << ".log";

                std::ofstream log(log_fname.str());
                log << "step\ttotal_gs\tcompute_gs\twrite_gs" << std::endl;
        #endif

                for (int it = restart_step; it < settings.steps;)
                {
        #ifdef ENABLE_TIMERS
                    MPI_Barrier(comm);
                    timer_total.start();
                    timer_compute.start();
        #endif

                    sim.iterate();
                    it++;

        #ifdef ENABLE_TIMERS
                    timer_compute.stop();
                    MPI_Barrier(comm);
                    timer_write.start();
        #endif

                    if (it % settings.plotgap == 0)
                    {
                        if (rank == 0)
                        {
                            std::cout << "Simulation at step " << it
                                      << " writing output step     "
                                      << it / settings.plotgap << std::endl;
                        }

                        writer_main.write(it, sim);
                    }

                    if (settings.checkpoint && (it % settings.checkpoint_freq)
        == 0)
                    {
                        WriteCkpt(comm, it, settings, sim, io_ckpt);
                    }

        #ifdef ENABLE_TIMERS
                    double time_write = timer_write.stop();
                    double time_step = timer_total.stop();
                    MPI_Barrier(comm);

                    log << it << "\t" << timer_total.elapsed() << "\t"
                        << timer_compute.elapsed() << "\t" <<
        timer_write.elapsed()
                        << std::endl;
        #endif
                }

                writer_main.close();
                */
    }
    Kokkos::finalize();
    /*
#ifdef ENABLE_TIMERS
    log << "total\t" << timer_total.elapsed() << "\t" << timer_compute.elapsed()
        << "\t" << timer_write.elapsed() << std::endl;

    log.close();
#endif
    */

    MPI_Finalize();
}
