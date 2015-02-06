#include "parameters.h"

Parameters::Parameters()
{
}

PatchesParameters::PatchesParameters(int init_patchSize)
{
    patchSize = init_patchSize;

}

int PatchesParameters::getPatchSize()
{
    return patchSize;
}

void PatchesParameters::setPatchSize( int init_patchSize )
{
    patchSize = init_patchSize;
}


//-----------------Parameters of pyramid---------------------------------------
PyramidParameters::PyramidParameters(int init_octaves, int init_octaveLayers,
                       double init_sigma )
{
    octaves = init_octaves;
    octaveLayers = init_octaveLayers;
    sigma = init_sigma;
}


int PyramidParameters::getOctavesNumber()
{
    return octaves;
}

int PyramidParameters::getOctaveLayersNumber()
{
    return octaveLayers;
}

double PyramidParameters::getSigmaValue()
{
    return sigma;
}

void PyramidParameters::setOctavesNumber( int init_octaves )
{
    octaves = init_octaves;
}

void PyramidParameters::setOctaveLayersNumber( int init_octaveLayers )
{
    octaveLayers = init_octaveLayers;
}

void PyramidParameters::setSigmaValue( double init_sigma )
{
    sigma = init_sigma;
}

void PyramidParameters::setParameters(int init_octaves, int init_octaveLayers, double init_sigma )
{
    octaves = init_octaves;
    octaveLayers = init_octaveLayers;
    sigma = init_sigma;
}

//---------------------------------------------------------------------------------------
