#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.18)
project(Gray-Scott-Kokkos C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS C)
find_package(Kokkos REQUIRED)
if(DEFINED Kokkos_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${Kokkos_CXX_COMPILER}")
    message("Setting the compiler to ${Kokkos_CXX_COMPILER}")
endif()

find_package(adios2 QUIET COMPONENTS CXX MPI)
if (adios2_FOUND)
    set(GS_HAS_ADIOS2 TRUE)
    message("Compiling with ADIOS2 support")
endif()

add_executable(gray-scott-kokkos
  main.cpp
  gray-scott.cpp
  settings.cpp
  restart.cpp
  writer.cpp
)
kokkos_compilation(SOURCE gray-scott.cpp)
if (adios2_FOUND)
    target_link_libraries(gray-scott-kokkos MPI::MPI_C Kokkos::kokkos)
else()
    target_link_libraries(gray-scott-kokkos MPI::MPI_C Kokkos::kokkos adios2::adios2 adios2::cxx11_mpi)
endif()
