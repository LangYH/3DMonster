#include "pyramid.h"

Pyramid::Pyramid(int init_octaves, int init_octaveLayers, double sigma)
{
    parameters = new PyramidParameters( init_octaves, init_octaveLayers, sigma );
}

Pyramid::~Pyramid()
{
    delete parameters;
}

void Pyramid::setParameters( int octaves, int octaveLayers, double sigma )
{
    parameters->setOctavesNumber( octaves );
    parameters->setOctaveLayersNumber( octaveLayers );
    parameters->setSigmaValue( sigma );
}

void Pyramid::buildGaussianPyramid(const Mat &base, std::vector<Mat> &pyramid)
{
    //get parameters
    int nOctaveLayers = parameters->getOctaveLayersNumber();
    int nOctaves = parameters->getOctavesNumber();
    double sigma = parameters->getSigmaValue();

    //pre-compute a series of sigma, which number is equal to nOctaveLayers
    std::vector<double> sig( nOctaveLayers );
    sig[0] = sigma;
    double k = std::pow( 2., 1./ nOctaveLayers );

    for( int i = 1; i < nOctaveLayers; i++ ){
        double sig_prev = std::pow( k, (double)(i-1) ) * sigma;
        double sig_total = sig_prev * k;
        sig[i] = std::sqrt( sig_total*sig_total - sig_prev * sig_prev );
    }

    //compute every layer of pyramid
    pyramid.clear();
    pyramid.resize( nOctaves * nOctaveLayers );
    for( int o = 0; o < nOctaves; o++ ){
        for( int l = 0; l < nOctaveLayers; l++ ){
            Mat& dst = pyramid[ o * nOctaveLayers + l ];
            if( o == 0 && l == 0)
                base.copyTo( dst );
                //GaussianBlur( base, dst, Size(), sig[0], sig[0]);
            else if( l == 0 ){
                const Mat& src = pyramid[ (o-1) * nOctaveLayers + nOctaveLayers - 1];
                GaussianBlur( src, dst, Size(), k * sig[nOctaveLayers - 1],
                        k * sig[nOctaveLayers - 1]);
                resize( dst, dst, Size( dst.cols/2, dst.rows/2 ),
                        0, 0, INTER_NEAREST );
            }
            else{
                const Mat& src = pyramid[ o * nOctaveLayers + l - 1];
                GaussianBlur( src, dst, Size(), sig[l], sig[l] );
            }
        }

    }

}
