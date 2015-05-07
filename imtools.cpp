#include "imtools.h"
#include <cmath>
#include <iostream>
#include "statistic.h"

imtools::imtools()
{
}

void imtools::matrixNormalize( Mat const &srcMat, Mat &dstMat )
{
    //Mat temp;
    //srcMat.convertTo( temp, CV_32FC1 );
    //double minVal, maxVal;
    //minMaxLoc( temp, &minVal, &maxVal );
    //dstMat = (temp-minVal) / ( maxVal - minVal );
    normalize( srcMat, dstMat, 0, 1.0, NORM_MINMAX, CV_32FC1 );
}

void imtools::weightsNormalize(Mat &matrix)
{
    matrix.convertTo( matrix, CV_32FC1 );
    matrix /= sum( matrix )[0];
}

void imtools::getImListAndDepthList( QString const &imPath, QString const &depthPath,
                          QStringList &imlist, QStringList &depthlist )
{
    QDir imDir( imPath );

    QStringList imFilters;
    imFilters += "*.png";

    foreach( QString imname, imDir.entryList( imFilters, QDir::Files ) ){
        imlist += imPath + QDir::separator() + imname;
        QStringList fields = imname.split( "_");
        depthlist += depthPath + QDir::separator() + "depth_"+ QFileInfo( fields[1] ).baseName() + ".yaml";
    }
}

void imtools::getDepthMapsWithIndexes( Mat const &indexes, QStringList const &depthlist, std::vector<Mat> &depthMaps)
{
    for( int i = 0; i < indexes.rows; i++ ){

        FileStorage fs( depthlist[indexes.at<ushort>(i,0)].toLocal8Bit().data(), FileStorage::READ );
        Mat depth;
        fs["depth"] >> depth;
        depthMaps.push_back( depth );
        fs.release();
    }
}

void imtools::fuseDepthMaps( std::vector<Mat> const &depthMaps, const Mat &weights, Mat &fusedDepthMap)
{
    fusedDepthMap.create( depthMaps[0].size(), CV_32FC1);
    fusedDepthMap = 0.0;

    for( int i = 0; i < int(depthMaps.size()); i++ ){
        Mat normalizedDepth;
        matrixNormalize( depthMaps[i], normalizedDepth );
        fusedDepthMap += normalizedDepth * weights.at<double>( i, 0 );
    }

}

void imtools::computeHOGDescriptorsMat(Mat &descriptorMat,
                                       const QStringList &imPath ,
                                       const HOGDescriptor *hogDesr)
{
    //compute all descriptors of training images
    //store in descriptorMat, for each row of it is a descriptor of one image
    descriptorMat.create( imPath.size(), hogDesr->getDescriptorSize(), CV_32FC1 );
    int nr = descriptorMat.rows;
    int nc = descriptorMat.cols;
    for( int j=0; j<nr; j++ ){
        float *data = descriptorMat.ptr<float>(j);
        Mat image = imread( imPath[j].toLocal8Bit().data(),
                            CV_LOAD_IMAGE_GRAYSCALE );
        std::vector<float> descr;
        hogDesr->compute( image, descr, Size(0, 0 ), Size( 0, 0 ) );
        for( int i = 0; i < nc; i++ ){
            *data++ = descr[i];
        }
    }

}

void imtools::getDepthMapsFromDepthlist(std::vector<Mat> &depthMaps, const QStringList &depthlist )
{
    FileStorage fs;
    foreach ( QString depthPath, depthlist ) {
        Mat depth;
        if( fs.open( depthPath.toLocal8Bit().data(), FileStorage::READ ) ){
            fs["depth"] >> depth;
            fs.release();
        }
        depthMaps.push_back( depth );
    }

}

double imtools::computeGradientEnergyWithHOG( const Mat &patch )
{

    double mean_value, deviation;
    Statistic::computeMeanAndDeviation( patch, mean_value, deviation );

    //return std::sqrt( gradient_energy );
    return deviation;

}
