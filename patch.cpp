#include "patch.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "imtools.h"
#include "statistic.h"
#include <iostream>
#include <QMessageBox>
#include "pyramid.h"
#include <sys/time.h>

Patch::Patch(int patchSize)
{
    parameters = new PatchesParameters( patchSize );
}

Patch::~Patch()
{
    delete parameters;
}

void Patch::randomSamplePatchesInPyramid(const Mat image, std::vector<Mat> &pyrs,
                                         std::vector< std::vector<Mat> > &patches_array,
                                         std::vector< std::vector<Point> > &coordinates_array,
                                         int octaves, int octaveLayers, double sigma,
                                         int number_patches, int number_of_target_layers )
{
    //pre-compute the number_vector, which indicate how many samples you want to samle
    //for each layers, the length of number_vector indicate how many layers you want sample
    std::vector<int> number_vector;
    prepareNumberVectorForPyramidSampling( number_vector, octaveLayers,
                                           number_patches, number_of_target_layers );

    //sample process
    randomSamplePatchesInPyramid( image, pyrs, patches_array, coordinates_array,
                                  octaves, octaveLayers, sigma, number_vector );

}

void Patch::prepareNumberVectorForPyramidSampling( std::vector<int> &number_vector,
                                                   int octaveLayers,
                                                   int number_patches, int number_of_target_layers )
{
    //pre-compute the number_vector, which indicate how many samples you want to samle
    //for each layers, the length of number_vector indicate how many layers you want sample
    std::vector<double> ratios( number_of_target_layers, 0.0 );
    double sum_of_ratios;
    for( int i = 0; i < number_of_target_layers; i++ ){
        double exponent = floor( i / double(octaveLayers) );
        ratios[i] = 1.0 / std::pow( 4., exponent );
        sum_of_ratios += ratios[i];
    }

    number_vector.assign( number_of_target_layers, 0 );
    for( int i = 0; i < number_of_target_layers; i++ ){
        number_vector[i] = int( ( ratios[i] / sum_of_ratios ) * number_patches );
    }

}

void Patch::randomSamplePatchesInPyramid(const Mat image, std::vector<Mat> &pyrs,
                                         std::vector< std::vector<Mat> > &patches_array,
                                         std::vector< std::vector<Point> > &coordinates_array,
                                         int octaves, int octaveLayers, double sigma,
                                         const std::vector<int> number_vector)
{
    Pyramid *imPyramid = new Pyramid( octaves, octaveLayers, sigma );

    imPyramid->buildGaussianPyramid( image, pyrs );

    randomSamplePatchesInMultipleMatrixes( pyrs, patches_array, coordinates_array, number_vector );

    delete imPyramid;
}

void Patch::randomSamplePatchesInMultipleMatrixes(const std::vector<Mat> &images,
                                                  std::vector< std::vector<Mat> > &patches_array,
                                                  std::vector< std::vector<Point> > &coordinates_array,
                                                  const std::vector<int> number_vector)
{
    CV_Assert( images.size() >= number_vector.size() );

    //clear up patches_array and coordinates_array
    std::vector< std::vector<Mat> >().swap( patches_array );
    std::vector< std::vector<Point> >().swap( coordinates_array );

    std::vector<Mat> patches_in_single_image;
    std::vector<Point> coordinates_in_single_image;
    for( unsigned int i = 0; i < number_vector.size(); i++ ){
        std::vector<Mat>().swap( patches_in_single_image );
        std::vector<Point>().swap( coordinates_in_single_image );
        randomSamplePatches( images[i], patches_in_single_image,
                             coordinates_in_single_image, number_vector[i] );
        patches_array.push_back( patches_in_single_image );
        coordinates_array.push_back( coordinates_in_single_image );
    }

}

void Patch::randomSamplePatches( Mat const &original_image , std::vector<Mat> &patches,
                          int nbr_patches )
{
    CV_Assert( original_image.rows > parameters->getPatchSize() &&
               original_image.cols > parameters->getPatchSize() );

    std::vector<Point> random_coordinates;

    generateRandomCoordinates(  original_image, random_coordinates, nbr_patches );

    samplePatches( original_image, random_coordinates, patches );
}

void Patch::randomSamplePatches(Mat const &original_image , std::vector<Mat> &patches,
                          std::vector<Point> &random_coordinates, int nbr_patches )
{
    generateRandomCoordinates(  original_image, random_coordinates, nbr_patches );

    samplePatches( original_image, random_coordinates, patches );
}


void Patch::generateRandomCoordinates( Mat const &original_image,
                                std::vector<Point> &random_coordinates, int nbr_patches )
{
    //clear up the std::vector<Point> first
    std::vector<Point>().swap( random_coordinates );

    struct timeval tv;
    struct timezone tz;
    gettimeofday( &tv, &tz );
    RNG rng( tv.tv_sec * 1000 + tv.tv_usec );
    int patchSize = parameters->getPatchSize();

    //define the downlimit and downlimit of sample area
    int downlimit_row = 0;
    int uplimit_row = original_image.rows - patchSize;
    int downlimit_column = 0;
    int uplimit_colomn = original_image.cols - patchSize;
    //**********************************************************
    for( int i = 0; i < nbr_patches ; i++ ){
        // y = [ downlimit_row, uplimit_row )
        // x = [ downlimit_column, uplimit_column )
        random_coordinates.push_back( Point( rng.uniform( downlimit_column, uplimit_colomn ),
                                             rng.uniform( downlimit_row, uplimit_row ) ) );
    }
}

void Patch::samplePatches( Mat const &original_image, std::vector<Point> const &random_coordinates,
                    std::vector<Mat> &patches )
{
    CV_Assert( original_image.cols > parameters->getPatchSize()
               && original_image.rows > parameters->getPatchSize() );
    //clean up the std::vector<Mat> first
    std::vector<Mat>().swap( patches );

    patches.resize( random_coordinates.size() );
    int patchSize = parameters->getPatchSize();
    for( unsigned int i = 0; i < random_coordinates.size(); i++){
        Mat roi( original_image, Rect( random_coordinates[i].x, random_coordinates[i].y,
                                       patchSize, patchSize ) );
        patches[i] = roi.clone();
    }
}

void Patch::detectOverlappedPatchesInPyramid( std::vector< std::vector<Mat> > &original_patches_array,
                                    std::vector< std::vector<PATCH_TYPE> > &symbols_array,
                                    double threshold, OVERLAP_METRIC metric)
{
    CV_Assert( !original_patches_array.empty() && !symbols_array.empty());

    //convert 2D array to 1D vector
    //and initialize symbols_array at the same time
    std::vector< std::vector<PATCH_TYPE> >().swap( symbols_array );
    symbols_array.assign( original_patches_array.size(), std::vector<PATCH_TYPE>() );
    std::vector<Mat> patches;
    for( unsigned int i = 0; i < original_patches_array.size(); i++ ){
        //[patches[0],patches[1],patches[2]...]
        patches.insert( patches.end(), original_patches_array[i].begin(),
                        original_patches_array[i].end() );
        symbols_array[i].assign( original_patches_array[i].size(), POSITIVE );
    }

    //detect ovelapped patches process
    std::vector<PATCH_TYPE> symbols;
    detectOverlappedPatches( patches, symbols, threshold, metric );

    //convert symbols result back to 2D array
    int index_symbols = 0;
    for( unsigned int i = 0; i < original_patches_array.size(); i++ ){
        for( unsigned int j = 0; j < original_patches_array[i].size(); j++ ){
            symbols_array[i][j] = symbols[ index_symbols++ ];
        }
    }

}

void Patch::detectFlatPatchesInPyramid( std::vector< std::vector<Mat> > &original_patches_array,
                                    std::vector< std::vector<PATCH_TYPE> > &symbols_array,
                                    double threshold, FLAT_METRIC metric)
{
    CV_Assert( !original_patches_array.empty() && !symbols_array.empty());

    //convert 2D array to 1D vector
    //and initialize symbols_array at the same time
    std::vector< std::vector<PATCH_TYPE> >().swap( symbols_array );
    symbols_array.assign( original_patches_array.size(), std::vector<PATCH_TYPE>() );
    std::vector<Mat> patches;
    for( unsigned int i = 0; i < original_patches_array.size(); i++ ){
        patches.insert( patches.end(), original_patches_array[i].begin(),
                        original_patches_array[i].end() );
        symbols_array[i].assign( original_patches_array[i].size(), POSITIVE );
    }

    //detect ovelapped patches process
    std::vector<PATCH_TYPE> symbols;
    detectFlatPatches( patches, symbols, threshold, metric );

    //convert symbols result back to 2D array
    int index_symbols = 0;
    for( unsigned int i = 0; i < original_patches_array.size(); i++ ){
        for( unsigned int j = 0; j < original_patches_array[i].size(); j++ ){
            symbols_array[i][j] = symbols[ index_symbols++ ];
        }
    }

}

void Patch::detectOverlappedPatches( std::vector<Mat> &original_patches,
                                    std::vector<PATCH_TYPE> &symbol,
                                    double threshold, OVERLAP_METRIC metric)
{
    if( threshold < 0 && threshold > 1 ){
        threshold = 0.8;
    }

    if( original_patches.empty() ){
        return;
    }

    //change all patches to gray scale
    std::vector<Mat> patches( original_patches.size() );
    for( unsigned int i = 0; i < original_patches.size(); i++ ){
        if( original_patches[i].channels() == 3 ){
            cvtColor( original_patches[i], patches[i], CV_BGR2GRAY );
        } else {
            patches[i] = original_patches[i].clone();
        }
    }

    //initialize symbols
    std::vector<PATCH_TYPE>().swap( symbol );
    symbol.assign( patches.size(), POSITIVE );

    //iterations:detect the overlapped patches
    std::vector<Mat>::iterator iter_patches = patches.begin();
    std::vector<PATCH_TYPE>::iterator iter_symbol = symbol.begin();
    for( ; iter_patches < patches.end() - 1 && iter_symbol < symbol.end() - 1;
         iter_patches++, iter_symbol++ ){
        //if the patch is already mark as the unuseful patch( which is 1 )
        //then jump over
        if( *iter_symbol != POSITIVE )
            continue;
        //for each patch, compute its cross correlation with every patch after it in vector patches
        std::vector<Mat>::iterator iter_patches_compared = iter_patches + 1;
        std::vector<PATCH_TYPE>::iterator iter_symbol_compared = iter_symbol  + 1;
        for( ; iter_patches_compared < patches.end() && iter_symbol_compared < symbol.end();
             iter_patches_compared++, iter_symbol_compared++ ){
            //if the patch is already labels with 1(means it overlaps with some patch)
            //then jump over
            if( *iter_symbol_compared != POSITIVE )
                continue;

            double score;
            if( metric == SSIM ){
                score = Statistic::computeCosineDistance( *iter_patches,
                                                             *iter_patches_compared );
            }else if( metric == CROSS_CORRELATION ){
                score = Statistic::computeCrossCorrelation( *iter_patches,
                                                 *iter_patches_compared );
            }
            if( score < threshold )
                *iter_symbol_compared = OVERLAP;
        }

    }

}


void Patch::detectFlatPatches( std::vector<Mat> &original_patches,
                                    std::vector<PATCH_TYPE> &symbol,
                              double threshold, FLAT_METRIC metric)
{
    if( threshold < 0 && threshold > 1 ){
        threshold = 0.8;
    }

    if( original_patches.empty() ){
        return;
    }

    //change all patches to gray scale
    std::vector<Mat> patches( original_patches.size() );
    for( unsigned int i = 0; i < original_patches.size(); i++ ){
        if( original_patches[i].channels() == 3 ){
            cvtColor( original_patches[i], patches[i], CV_BGR2GRAY );
        } else {
            patches[i] = original_patches[i].clone();
        }
    }

    //initialize symbol
    std::vector<PATCH_TYPE>().swap( symbol );
    symbol.assign( original_patches.size(), POSITIVE );

    //iterations:detect the flat patches, using gradient energy
    std::vector<Mat>::iterator iter_patches = patches.begin();
    std::vector<PATCH_TYPE>::iterator iter_symbol = symbol.begin();
    for( ; iter_patches < patches.end() - 1 && iter_symbol < symbol.end() - 1;
         iter_patches++, iter_symbol++ ){
        //if the patch is already mark as the unuseful patch( which is 1 )
        //then jump over
        if( *iter_symbol != POSITIVE )
            continue;

        //compute gradient energy score
        double score;
        if( metric == DEVIATION )
            score = imtools::computeGradientEnergyWithHOG( *iter_patches );

        if( score < threshold )
            *iter_symbol = FLAT;

    }

}

void Patch::drawFrameForPatchesInImage( Mat &image, const std::vector<Point> coordinates,
                                        const std::vector<PATCH_TYPE> overlappedPatchSymbols,
                                        const std::vector<PATCH_TYPE> flatPatchSymbols,
                                        bool showPositives, bool showOverlappes, bool showFlats,
                                        const int patchesToShow,  const int patchSize )
{
    CV_Assert( patchesToShow <= int( coordinates.size() ) );
    CV_Assert( image.cols > patchSize && image.rows > patchSize );
    CV_Assert( coordinates.size() == overlappedPatchSymbols.size()
               && coordinates.size() == flatPatchSymbols.size() );

    for( int i = 0; i < patchesToShow; i++ ){
        if( showOverlappes && overlappedPatchSymbols[i] == OVERLAP ){
            //show ovelapped patches in red frame
            rectangle( image,
                       Rect(coordinates[i].x, coordinates[i].y, patchSize, patchSize ),
                       Scalar( 0, 0, 255 ), 1, 8 );
        }

        else if( showFlats && flatPatchSymbols[i] == FLAT ){
            //show flat pathes in blue frame
            rectangle( image,
                       Rect(coordinates[i].x, coordinates[i].y, patchSize, patchSize ),
                       Scalar( 255, 0, 0 ), 1, 8 );
        }

        else if( showPositives && overlappedPatchSymbols[i] == POSITIVE
                 && flatPatchSymbols[i] == POSITIVE ){
            //show positive patches in green frame
            rectangle( image,
                       Rect(coordinates[i].x, coordinates[i].y, patchSize, patchSize ),
                       Scalar( 0, 255, 0 ), 1, 8 );
        }
    }
}
