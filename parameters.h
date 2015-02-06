#ifndef PARAMETERS_H
#define PARAMETERS_H

class Parameters
{
public:
    Parameters();
};

class PatchesParameters
{
public:
    PatchesParameters( int init_patchSize);

    int getPatchSize();

    void setPatchSize(int init_patchSize );

private:
    int patchSize;
};

class PyramidParameters
{
public:
    PyramidParameters(int init_octaves, int init_octaveLayers, double init_sigma);

    int getOctavesNumber();
    int getOctaveLayersNumber();
    double getSigmaValue();

    void setOctavesNumber( int init_octaves );
    void setOctaveLayersNumber( int init_octaveLayers );
    void setSigmaValue( double init_sigma );
    void setParameters(int init_octaves, int init_octaveLayers, double init_sigma );

private:
    int octaves;
    int octaveLayers;
    int totalLayers;
    double sigma;
};

#endif // PARAMETERS_H
