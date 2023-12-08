# Performance for Gray-Scott simulation

We use TAU and the profiling mechanism in ADIOS2 to gather performance characteristics for the Gray-Scott simulation.

The Kernels, GS-Kokkos version spends most of the time are the following:
```
100.0        0.185       23,233           1           1   23233810 .TAU application
100.0        2,277       23,233           1       53894   23233625 int taupreload_main(int, char **, char **)
 72.1       16,751       16,751        1000           1      16751 Kokkos::parallel_for calc_gray_scott [type = HIP, device = 0]
  7.1        1,648        1,650         101        2230      16339 BP5Writer::EndStep
  2.9          470          676         303         202       2232 void adios2::format::BP5Serializer::Marshal(void*, const char*, adios2::DataType, std::size_t, std::size_t, const size_t*, const size_t*, const size_t*, const void*, bool, adios2::format::BufferV::BufferPos*)
  1.5          342          342       12000           0         29 MPI_Sendrecv()
  1.5          342          342           1           0     342408 MPI_Init_thread()
  1.5          341          341       16000           0         21 Kokkos::parallel_for Kokkos::ViewCopy-2D [type = HIP, device = 0]
  1.3          295          295           2           0     147996 Kokkos::parallel_for Kokkos::ViewFill-1D [type = HIP, device = 0]
  0.9            1          206         202         202       1021 void adios2::format::GetMinMax(const void*, std::size_t, adios2::DataType, adios2::MinMaxStruct&, adios2::MemorySpace)
```

The CPU version has the following profile:
```
100.0        0.174     1:04.251           1           1   64251713 .TAU application
100.0     1:00.333     1:04.251           1       12490   64251539 int taupreload_main(int, char **, char **)
  2.5        1,599        1,600         101        2230      15850 BP5Writer::EndStep
  1.6        1,004        1,004       12000           0         84 MPI_Sendrecv()
  1.4            1          902         303         202       2977 void adios2::format::BP5Serializer::Marshal(void*, const char*, adios2::DataType, std::size_t, std::size_t, const size_t*, const size_t*, const size_t*, const void*, bool, adios2::format::BufferV::BufferPos*)
  1.4          901          901         202           0       4460 void adios2::format::GetMinMax(const void*, std::size_t, adios2::DataType, adios2::MinMaxStruct&, adios2::MemorySpace)
```

Perfromance on one node on Frontier (problem size of 64x64x64)

<img width="361" alt="image" src="https://github.com/anagainaru/Gray-Scott-Kokkos/assets/16229479/e0cb3695-39f7-4971-86ff-26731b898a7a">


**Using TAU to profile the code**

Example TAU trace using one node and 8 processes on Frontier
```
---------------------------------------------------------------------------------------
%Time    Exclusive    Inclusive       #Call      #Subrs  Inclusive Name
              msec   total msec                          usec/call
---------------------------------------------------------------------------------------
100.0        0.244        3,532           1           1    3532007 .TAU application
100.0        1,332        3,531           1       47894    3531763 int taupreload_main(int, char **, char **)
 28.6        1,010        1,010        1000           1       1011 Kokkos::parallel_for calc_gray_scott [type = HIP, device = 0]
  9.7          282          343         101        4048       3398 BP5Writer::EndStep
  6.8          241          241       16000           0         15 Kokkos::parallel_for Kokkos::ViewCopy-2D [type = HIP, device = 0]
  5.0           37          175         303         202        579 void adios2::format::BP5Serializer::Marshal(void*, const char*, adios2::DataType, std::size_t, std::size_t, const size_t*, const size_t*, const size_t*, const void*, bool, adios2::format::BufferV::BufferPos*)
  3.9        0.889          138         202         202        685 void adios2::format::GetMinMax(const void*, std::size_t, adios2::DataType, adios2::MinMaxStruct&, adios2::MemorySpace)
  3.9          137          137         202           0        681 Kokkos::parallel_reduce N6Kokkos4Impl31CombinedReductionFunctorWrapperIZN12_GLOBAL__N_116KokkosMinMaxImplIdEEvPKT_mRS4_S7_EUliRdS8_E_NS_9HostSpaceEJNS_3MaxIdSA_EENS_3MinIdSA_EEEEE [type = HIP, device = 0]
  3.2          113          113           2           0      56646 Kokkos::parallel_for Kokkos::ViewFill-1D [type = HIP, device = 0]
  2.1           75           75       12000           0          6 MPI_Sendrecv()
  1.9           68           68        6000           0         11 Kokkos::parallel_for Kokkos::View::initialization [firstGhostPlane] via memset [type = HIP, device = 0]
  1.8           64           64        6000           0         11 Kokkos::parallel_for Kokkos::View::initialization [lastGhostPlane] via memset [type = HIP, device = 0]
  1.8           63           63        6000           0         11 Kokkos::parallel_for Kokkos::View::initialization [temp] via memset [type = HIP, device = 0]
  1.5           51           51         202           0        256 MPI_Win_free()
```

**ADIOS2 profiling**

Example profiling log gathered by ADIOS2 for two ranks (where only rank 0 does writing)
```
{ "rank":0, "start":"Wed_Dec_06_10:53:10_2023","ES_meta1_gather_mus": 1198, "ES_meta1_gather":{"mus":1198, "nCalls":100},"ES_mus": 357129, "ES":{"mus":357129, "nCalls":100},"Marshal_mus": 189057, "Marshal":{"mus":189057, "nCalls":300},"ES_meta1_mus": 1824, "ES_meta1":{"mus":1824, "nCalls":100},"ES_meta2_mus": 3190, "ES_meta2":{"mus":3190, "nCalls":100},"ES_close_mus": 1126, "ES_close":{"mus":1126, "nCalls":100},"ES_AWD_mus": 350717, "ES_AWD":{"mus":350717, "nCalls":100}, "databytes":0, "metadatabytes":0, "metametadatabytes":0, "transport_0":{"type":"File_POSIX", "wbytes":419430400, "close":{"mus":444, "nCalls":1}, "write":{"mus":233151, "nCalls":400}, "open":{"mus":1654, "nCalls":1}}, "transport_1":{"type":"File_POSIX", "wbytes":178720, "close":{"mus":364, "nCalls":1}, "write":{"mus":1807, "nCalls":704}, "open":{"mus":831, "nCalls":1}} },
{ "rank":1, "start":"Wed_Dec_06_10:53:10_2023","ES_meta1_gather_mus": 248, "ES_meta1_gather":{"mus":248, "nCalls":100},"ES_mus": 355382, "ES":{"mus":355382, "nCalls":100},"Marshal_mus": 190353, "Marshal":{"mus":190353, "nCalls":300},"ES_meta1_mus": 431, "ES_meta1":{"mus":431, "nCalls":100},"ES_meta2_mus": 0, "ES_meta2":{"mus":0, "nCalls":100},"ES_close_mus": 739, "ES_close":{"mus":739, "nCalls":100},"ES_AWD_mus": 353988, "ES_AWD":{"mus":353988, "nCalls":100}, "databytes":0, "metadatabytes":0, "metametadatabytes":0 },
```

ES includes all the logic in EndStep which includes the following:
* ES_close
Measures the time to m_BP5Serializer.CloseTimestep

* ES_AWD is the heavyweight part of EndStep, that calls the WriteData routine to move data to storage

* ES_meta1 is the time for the metameta gathering (included in ES_meta1_gather) and the write time.

* ES_meta2 is the time for the metadata (min/max) gathering (included in ES_meta2_gather) and the write time.

* Marshal copies data to internal buffers and computes the min/max
