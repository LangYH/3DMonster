#include "imtools.h"
#include <cmath>
#include <iostream>
#include "statistic.h"

#define Cutoff ( 3 )


typedef struct SortingEntry{
    int index;
    ElementType elem;
} SortingEntry, *SortingEntryPtr;

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

void Swap( SortingEntryPtr *a, SortingEntryPtr *b )
{
    SortingEntryPtr tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

void insertSort( SortingEntryPtr A[], int N )
{
    int j, p;

    SortingEntryPtr tmp;
    for( p = 1; p < N; p++ ){
        tmp = A[p];
        for( j = p; j > 0 && A[j-1]->elem > tmp->elem; j-- )
            A[j] = A[j-1];
        A[j] = tmp;
    }
}

ElementType median3( SortingEntryPtr A[], int left, int right )
{
    int center = ( left + right ) / 2;

    if( A[left]->elem > A[center]->elem ){
        Swap( &A[left], &A[center] );
    }
    if( A[left]->elem > A[right]->elem ){
        Swap( &A[left], &A[right] );
    }
    if( A[center]->elem > A[right]->elem ){
        Swap( &A[center], &A[right] );
    }

    Swap( &A[center], &A[right-1] );

    return A[right-1]->elem;
}

void Qsort( SortingEntryPtr A[], int left, int right )
{
    int i, j;
    ElementType pivot;

    if( left + Cutoff <= right )
    {
        pivot = median3( A, left, right );
        i = left;
        j = right - 1;
        for(;;){
            while( A[++i]->elem < pivot )
                ;
            while( A[--j]->elem > pivot )
                ;
            if( i < j ){
                Swap( &A[i], &A[j] );
            }
            else
                break;
        }
        Swap( &A[i], &A[right-1] );

        Qsort( A, left, i - 1 );
        Qsort( A, i + 1, right );
    }
    else{
        //insert sort
        insertSort( A + left, right - left + 1 );

    }
}

void imtools::idxSort( ElementType Data[], int SortedIndex[], int N )
{
    SortingEntryPtr A[N];
    for( int i = 0; i < N; i++ ){
        A[i] = (SortingEntryPtr)malloc( sizeof( SortingEntry ) );
        if( A[i] == NULL ){
            printf( "No space for sorting" );
            exit(1);
        }
        A[i]->elem = Data[i];
        A[i]->index = i;
    }

    Qsort( A, 0, N - 1 );

    for( int i = 0; i < N; i++ ){
        SortedIndex[i] = A[i]->index;
        free( A[i] );
    }
}
