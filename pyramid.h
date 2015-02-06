#ifndef PYRAMID_H
#define PYRAMID_H
#include "parameters.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

class Pyramid
{
public:
    Pyramid( int init_octaves, int init_octaveLayers, double sigma );
    ~Pyramid();

private:

public:
    PyramidParameters *parameters;
    void buildGaussianPyramid( const Mat &base,
                               std::vector<Mat> &pyramid );
    void setParameters( int octaves, int octaveLayers, double sigma );
};

#endif // PYRAMID_H
