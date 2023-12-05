#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.12)
project(Gray-Scott-Kokkos)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS C)
find_package(Kokkos REQUIRED)
if(DEFINED Kokkos_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${Kokkos_CXX_COMPILER}")
    message("Setting the compiler to ${Kokkos_CXX_COMPILER}")
endif()

find_package(ADIOS2 QUIET COMPONENTS CXX MPI)
if (adios2_FOUND)
    set(GS_HAS_ADIOS2 TRUE)
    message("Compiling with ADIOS2 support")
endif()

add_executable(gray-scott-kokkos
  main.cpp
  settings.cpp
  writer.cpp
)
kokkos_compilation(SOURCE gray-scott.cpp)
kokkos_compilation(SOURCE main.cpp)
if (adios2_FOUND)
    target_link_libraries(gray-scott-kokkos MPI::MPI_C Kokkos::kokkos)
else()
    target_link_libraries(gray-scott-kokkos MPI::MPI_C Kokkos::kokkos adios2::cxx11_mpi)
endif()
