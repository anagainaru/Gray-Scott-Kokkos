#ifndef __GS_SETTINGS_H__
#define __GS_SETTINGS_H__

#include <fstream>
#include <string>

#include "json.hpp"

namespace grayscott
{
struct Settings
{
    size_t L;
    int steps;
    int output_gap;
    double F;
    double k;
    double dt;
    double Du;
    double Dv;
    double noise;
    std::string output;
    std::string adios_config;

    Settings(const std::string &fname)
    {
        std::ifstream ifs(fname);
        nlohmann::json j;
        ifs >> j;
        L = j.at("L");
        steps = j.at("steps");
        output_gap = j.at("output_gap");
        F = j.at("F");
        k = j.at("k");
        dt = j.at("dt");
        Du = j.at("Du");
        Dv = j.at("Dv");
        noise = j.at("noise");
        output = j.at("output");
        adios_config = j.at("adios_config");
    }

    void print() const
    {
        std::cout << "grid:             " << L << "x" << L << "x" << L
                  << std::endl;
        std::cout << "steps:            " << steps << std::endl;
        std::cout << "output_gap:       " << output_gap << std::endl;
        std::cout << "F:                " << F << std::endl;
        std::cout << "k:                " << k << std::endl;
        std::cout << "dt:               " << dt << std::endl;
        std::cout << "Du:               " << Du << std::endl;
        std::cout << "Dv:               " << Dv << std::endl;
        std::cout << "noise:            " << noise << std::endl;
        std::cout << "output:           " << output << std::endl;
        std::cout << "adios_config:     " << adios_config << std::endl;
    }
};
}
#endif
