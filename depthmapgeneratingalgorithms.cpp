#include "depthmapgeneratingalgorithms.h"
#include "lineextractingalgorithms.h"
#include <QStringList>
#include <filters.h>
#include <imtools.h>
#include <knnsearcher.h>
#include <kmeanssearcher.h>
#define LINE_COUNTS 51

#include <math.h>
#include <iostream>
#include <vector>

DepthMapGeneratingAlgorithms::DepthMapGeneratingAlgorithms()
{
}

Mat DepthMapGeneratingAlgorithms::usingRelativeHeightDepthCue( Mat const &inputImage )
{
    if( inputImage.empty() )
    {
        Mat emptyImage;
        return emptyImage;
    }
    Mat image, edgeMap;
    inputImage.copyTo( image );

    //Mat edgeMap = LineExtractingAlgorithms::canny( image );
    cv::Size originalImageSize = image.size();
    cv::GaussianBlur( image, image, cv::Size( 5, 5 ), 0, 0, BORDER_DEFAULT );
    cv::resize( image, image, cv::Size( int( image.cols * ( 250.0 / image.rows) ), 250 ) );
    cv::Canny( image, edgeMap,60, 180 );

    Mat lineMap = lineTracing( edgeMap );

    Mat depthMap = createDepthMapUsingLineMap( edgeMap, lineMap );

    cv::resize( depthMap, depthMap, originalImageSize );

    return depthMap;
}

Mat DepthMapGeneratingAlgorithms::lineTracing( Mat const &edgeMap )
{
    double a = 0.1, b = edgeMap.rows / 4, pc = edgeMap.rows / 4;
    double alpha = 0.4, beta = 0.3, gama = 0.3;

    Mat lineMap( LINE_COUNTS, edgeMap.cols, CV_8U, Scalar(0) );
    //initialize lineMap ,assign row number to the first colume of lineMap
    //      0 0 0 0 0 0 ...
    //      15 0 0 0 0 0...
    //      30 0 0 0 0 0...
    for( int i = 0; i < LINE_COUNTS; i++ )
    {
        lineMap.at<uchar>( i, 0 ) = cvCeil( double( edgeMap.rows )/LINE_COUNTS * i );
    }

    //starting line tracing...
    //lineMap.row( 0 ) = 0; //top border line
    lineMap.row( lineMap.rows-1 ) = edgeMap.rows-1;//bottom border line
    for( int r = 1; r < LINE_COUNTS-1; r++ )
    {
        for( int c = 1; c < edgeMap.cols; c++ )
        {
            //for each point of lineMap( except the first column )
            Mat constraintValueColumn( edgeMap.rows, 1, CV_64F, 0.0 );
            for( int rc = 0; rc < edgeMap.rows; rc++ )
            {
                //edge value constrant
                int y1 = lineMap.at<uchar>( r, c-1 );//current position
                int y2 = lineMap.at<uchar>( r, 0 );//starting position
                double Elt = std::exp( -double(edgeMap.at<uchar>( rc, c )) / a );
                double Es = double( std::abs( rc - y1) ) / b;
                double Ee = double( std::abs( rc - y2) ) / pc;
                constraintValueColumn.at<double>( rc, 0 ) = alpha * Elt + beta * Es + gama * Ee;
            }
            //find the minimal value in constraintValueColumn
            cv::Point minLoc;
            cv::minMaxLoc( constraintValueColumn, NULL, NULL, &minLoc, NULL );
            lineMap.at<uchar>( r, c ) = minLoc.y;
        }
    }

    return lineMap;

}


Mat DepthMapGeneratingAlgorithms::createDepthMapUsingLineMap( Mat const &edgeMap, Mat const &lineMap )
{
    Mat depthMap( edgeMap.size(), CV_8U, Scalar(0) );
    for( int i = 1; i<lineMap.rows; i++ )
    {
        for( int j = 0; j < lineMap.cols; j++ )
        {
            //depthMap.col(j).rowRange( cv::Range( lineMap.at<uchar>( i-1, j ), lineMap.at<uchar>(i, j)) - 1 ) = i * 5 ;
            for( int point = lineMap.at<uchar>( i-1, j ); point < lineMap.at<uchar>(i, j) ; point++ )
                depthMap.at<uchar>( point, j ) = 250 - ( i - 1) * 5;
        }
    }
    cv::GaussianBlur( depthMap, depthMap, cv::Size( 3, 3 ), 0, 0, BORDER_DEFAULT );
    return depthMap;
}

void DepthMapGeneratingAlgorithms::usingkNNWithHOG( Mat const &inputImage, Mat &finalDepthMap, int k )
{
    if( inputImage.empty() ){
        std::cout << "no input image" << std::endl;
        return;
    }
    QStringList imlist;
    QStringList depthlist;

    QString imPath = "/home/lang/dataset/NYUDataset/images";
    QString depthPath = "/home/lang/dataset/NYUDataset/depthInYAML";

    imtools::getImListAndDepthList( imPath, depthPath, imlist, depthlist );
    if( imlist.empty() || depthlist.empty() ){
        std::cout << "imlist or depthlist is empty!" << std::endl;
        return;
    }

    //------------------kNN search with HOG----------------------
    Mat indexes; //column vector, containing the top k indexes
    Mat weights; //column vector, normalized HOG distance from the target to the k images
    std::cout << "kNN searching.." << std::endl;
    kNNSearcher::kNNSearchWithHOG( inputImage, imlist, indexes, weights, k );
    //-----------------------------------------------------------

    //--------------get correspondent depths-----------------------
    std::cout << "getting depthMaps from depthlist..." << std::endl;
    std::vector<Mat> depthMaps;
    imtools::getDepthMapsWithIndexes( indexes, depthlist, depthMaps );
    if( depthMaps.empty() ){
        std::cout << "depthMaps is empty!!" << std::cout;
        return;
    }
    //------------------------------------------------------------

    //----------------depths fused-------------------------------
    std::cout << "fusing depthMap" << std::endl;
    Mat fusedDepthMap;
    imtools::fuseDepthMaps( depthMaps, weights, fusedDepthMap );
    //------------------------------------------------------------

    //----------------depthmap filtering--------------------------
    std::cout << "transfomation finished!" << std::endl;
    double minVal, maxVal;
    minMaxLoc( fusedDepthMap, &minVal, &maxVal );
    double s = 255.0 / maxVal;
    fusedDepthMap *= s;
    convertScaleAbs( fusedDepthMap, fusedDepthMap );
    fusedDepthMap.copyTo( finalDepthMap );
    //------------------------------------------------------------

}

void DepthMapGeneratingAlgorithms::usingkmeansWithHOG( Mat const &inputImage, Mat &finalDepthMap, int k )
{
    if( inputImage.empty() ){
        std::cout << "no input image" << std::endl;
        return;
    }
    QStringList imlist;
    QStringList depthlist;

    QString imPath = "/home/lang/dataset/NYUDataset/images";
    QString depthPath = "/home/lang/dataset/NYUDataset/depthInYAML";

    imtools::getImListAndDepthList( imPath, depthPath, imlist, depthlist );
    if( imlist.empty() || depthlist.empty() ){
        std::cout << "imlist or depthlist is empty!" << std::endl;
    }

    //------------------kmeans search with HOG----------------------
    std::cout << "kmeans classifying" << std::endl;
    kmeansSearcher searcher( imlist, depthlist, k );
    searcher.train();
    Mat depthMap;

    //-----------------------------------------------------------

}



