#include "depthmakerwithvisualword.h"
#include "filters.h"
#include "patch.h"
#include "pyramid.h"
#include "imtools.h"
#include "sys/time.h"

DepthMakerWithVisualWord::DepthMakerWithVisualWord()
{
    dict = new VisualWordDictionary();
}

DepthMakerWithVisualWord::~DepthMakerWithVisualWord()
{
    delete dict;
}

Mat DepthMakerWithVisualWord::generateDepthmapInMultiScale(const Mat image)
{

}

void DepthMakerWithVisualWord::searchImageForGivenWord(const Mat &image, MySVM &svm,
                                                       std::vector<Rect> &filtered_result,
                                                       std::vector<double> &filtered_scores )
{
    std::vector<float> myDetector;
    VisualWordDictionary *dict = new VisualWordDictionary();
    dict->getSVMDetectorForHOG( &svm, myDetector );

    HOGDescriptor myHOG(Size(80,80),Size(8,8),Size(8,8), Size(8,8), 9 );
    myHOG.setSVMDetector( myDetector );

    std::vector<Rect> found;
    std::vector<double> scores;

    //struct timeval tv;
    //gettimeofday( &tv, NULL );
    myHOG.detectMultiScale( image, found, scores, -0.8, Size(8,8), Size(0,0), 1.3, 2 );
    //struct timeval tv1;
    //gettimeofday( &tv1, NULL );
    //std::cout << "detectmultiscal elapsed with " << tv1.tv_sec - tv.tv_sec
    //          << " seconds " << ( tv1.tv_usec - tv.tv_usec ) / 1000000.0 <<  " ms "
    //          << std::endl;

    for( unsigned i = 0; i < found.size(); i++ ){
        Rect r = found[i];
        if( r.x < 0 || r.y < 0 || ( r.x + r.width ) > image.cols
                || ( r.y + r.height ) > image.rows ){
            continue;
        }
        unsigned int j = 0;
        for( ; j < found.size(); j++ )
            if( j != i && ( r & found[j] ) == r )
                break;

        if( j == found.size() ){
            filtered_result.push_back( r );
            filtered_scores.push_back( scores[i]);
        }
    }
}

void DepthMakerWithVisualWord::generateDepthmap(const Mat &image,
                                                std::vector<Mat> &results )
{

    std::vector<unsigned> nbr_vector;
    nbr_vector.push_back( 10 );
    nbr_vector.push_back( 3 );


    std::vector<Mat> pyrs;
    std::vector< std::vector<Mat> > patches_array;
    std::vector< std::vector<Point> > coordinates_array;

    samplePatchesForDepthGeneration( image, pyrs, patches_array, coordinates_array );

    std::vector< std::vector< int > > class_array;
    std::vector< std::vector< double > > score_array;
    std::vector< std::vector< int > > sorted_index_array;
    classifyAllPatches( patches_array, class_array, score_array );
    sortedClassifiedResults( score_array, sorted_index_array );

    std::vector<Mat> depths;
    generateInitDepthMap( pyrs, depths );
    for( unsigned int i = 0; i < sorted_index_array.size(); i++ ){
        for(  unsigned j = 0; j < nbr_vector[i] && j < sorted_index_array[i].size(); j++ ){
            int tail = nbr_vector[i] < sorted_index_array[i].size() ?
                                    nbr_vector[i] : sorted_index_array[i].size();
            int current_index = sorted_index_array[i][ tail - j - 1];
            if( score_array[i][current_index] > -0.5 ){
                //get the corresponding depthmap
                Mat depthmap;
                depthmap = dict->searchForDepthWithGiveID( class_array[i][current_index] );
                //Filters filters;
                //filters.guidedFilter( depthmap, patches_array[i][current_index],
                //                      depthmap, 30, 0.1 );
                depthmap.convertTo(depthmap, CV_32FC1 );

                normalize( depthmap, depthmap, 0, 1.0, NORM_MINMAX, CV_32FC1 );
                double ratio = ( double( depths[i].rows ) - coordinates_array[i][current_index].y ) /
                                depths[i].rows * 255.0;
                Mat temp = depthmap * ratio;
                temp.convertTo( depthmap, CV_8UC1 );

                if( depths[i].channels() == 3 && depthmap.channels() == 1 )
                    cvtColor( depthmap, depthmap, CV_GRAY2BGR );

                Mat imageROI = depths[i]( cv::Rect( coordinates_array[i][current_index].x,
                                                    coordinates_array[i][current_index].y,
                                                    depthmap.cols, depthmap.rows ) );
                cv::addWeighted( imageROI, 0., depthmap, 1.0, 0., imageROI );

            }
        }
    }

    for( unsigned i = 0; i < depths.size(); i++ ){
        Mat elarge_image;

        Mat temp;
        Filters filters;
        filters.guidedFilter( depths[i], pyrs[i],
                              temp, 45, 0.1 );

        cv::resize( temp, elarge_image, image.size() );

        results.push_back( elarge_image );
    }


}

void DepthMakerWithVisualWord::generateInitDepthMap(const std::vector<Mat> &images, std::vector<Mat> &depths)
{
    std::vector<Mat>().swap( depths );
    for( unsigned i = 0; i < images.size(); i++ ){
        Mat depth( images[i].rows, images[i].cols, CV_8UC1, Scalar(0) );
        int nr = depth.rows;
        int nc = depth.cols * depth.channels();
        for( int i = 0; i < nr; i++ ){
            uchar* data = depth.ptr<uchar>(i);
            for( int j = 0; j < nc; j++ ){
                *data++ = int( double( nr - i ) / double(nr) * 255.0 ) ;
            }
        }
        depths.push_back( depth );
    }
}

void DepthMakerWithVisualWord::initialDepthMap(const std::vector<Mat> &images, std::vector<Mat> &depths)
{
    std::vector<Mat>().swap( depths );
    for( unsigned i = 0; i < images.size(); i++ ){
        Mat depth = images[i].clone();
        depths.push_back( depth );
    }
}

void DepthMakerWithVisualWord::samplePatchesForDepthGeneration(const Mat &image, std::vector<Mat> &pyrs, std::vector<std::vector<Mat> > &patches, std::vector<std::vector<Point> > &coordinates)
{
    std::vector< std::vector< Mat > >().swap( patches );
    std::vector< std::vector< Point > >().swap( coordinates );

    Patch *patchExtracter;
    patchExtracter = new Patch( 80 );


    //drop the first two layers of pyramid
    Pyramid imPyramid( 3, 2, 1.0 );
    imPyramid.buildGaussianPyramid( image, pyrs );
    pyrs.erase( pyrs.begin() );
    pyrs.erase( pyrs.begin() );
    pyrs.erase( pyrs.begin() + 1 );
    pyrs.erase( pyrs.begin() + 1 );

    std::vector<int> number_vector;
    number_vector.push_back( 150 );
    number_vector.push_back( 150 );

    //sampling patches
    std::vector< std::vector<Mat> > patches_array;
    std::vector< std::vector<Point> > coordinates_array;
    std::vector<Mat> patches_in_single_image;
    std::vector<Point> coordinates_in_single_image;
    for( unsigned int i = 0; i < pyrs.size(); i++ ){
        std::vector<Mat>().swap( patches_in_single_image );
        std::vector<Point>().swap( coordinates_in_single_image );
        patchExtracter->randomSamplePatches( pyrs[i], patches_in_single_image,
                                             coordinates_in_single_image, number_vector[i] );
        patches_array.push_back( patches_in_single_image );
        coordinates_array.push_back( coordinates_in_single_image );
    }

    //intialize overlap symbols and flatPatchSymbols
    std::vector< std::vector<PATCH_TYPE> > overlappedPatchSymbols_array;
    std::vector< std::vector<PATCH_TYPE> > flatPatchSymbols_array;
    overlappedPatchSymbols_array.assign( patches_array.size(), std::vector<PATCH_TYPE>() );
    flatPatchSymbols_array.assign( patches_array.size(), std::vector<PATCH_TYPE>() );
    for( unsigned int i = 0; i < patches_array.size(); i++ ){
        overlappedPatchSymbols_array[i].assign( patches_array[i].size(), POSITIVE );
        flatPatchSymbols_array[i].assign( patches_array[i].size(), POSITIVE );
    }

    //remove overlapped and flat patches
    patchExtracter->detectOverlappedPatchesInPyramid( patches_array,
                                                      overlappedPatchSymbols_array,
                                                      60,
                                                      CROSS_CORRELATION );

    patchExtracter->detectFlatPatchesInPyramid( patches_array,
                                                flatPatchSymbols_array,
                                                10.0,
                                                DEVIATION );

    //return only the positive patches
    patches.assign( patches_array.size(), std::vector<Mat>() );
    coordinates.assign( coordinates_array.size(), std::vector<Point>() );
    for( unsigned i = 0; i < patches_array.size(); i++ ){
        for( unsigned j = 0; j < patches_array[i].size(); j++ ){
            if( overlappedPatchSymbols_array[i][j] == POSITIVE &&
                    flatPatchSymbols_array[i][j] == POSITIVE ){
                patches.at(i).push_back( patches_array[i][j] );
                coordinates.at(i).push_back( coordinates_array[i][j] );
            }
        }
    }

    delete patchExtracter;

}

void DepthMakerWithVisualWord::sortedClassifiedResults(std::vector<std::vector<double> > score_array, std::vector<std::vector<int> > &sorted_index)
{
    std::vector< std::vector< int > >().swap( sorted_index );
    sorted_index.assign( score_array.size(), std::vector< int >() );

    for( unsigned i = 0; i < score_array.size(); i++ ){
        imtools::idxSort( score_array[i], sorted_index[i], true );
    }

}

void DepthMakerWithVisualWord::classifyAllPatches(const std::vector<std::vector<Mat> > patches_array, std::vector<std::vector<int> > &result_class, std::vector<std::vector<double> > &svm_score)
{
    //initialize result_class and svm_score
    std::vector< std::vector< int > >().swap( result_class );
    std::vector< std::vector< double > >().swap( svm_score );
    result_class.assign( patches_array.size(), std::vector<int>() );
    svm_score.assign( patches_array.size(), std::vector<double>() );
    for( unsigned i = 0; i < patches_array.size(); i++ ){
        result_class[i].assign( patches_array[i].size(), -1 );
        svm_score[i].assign( patches_array[i].size(), -10.0 );
    }

    //load all the SVM classifier
    std::map<int, CvSVM*> classifiers;
    VisualWord::loadAllSVMClassifiers( classifiers );

    //classify all patches
    for( unsigned int i = 0; i < patches_array.size(); i++ ){
        for(  unsigned int j = 0; j < patches_array[i].size(); j++ ){
            double best_match_score = -10.0;
            int best_match_class_id = dict->searchForId( patches_array[i][j], best_match_score);

            result_class[i][j] = best_match_class_id;
            svm_score[i][j] = best_match_score;
        }
    }
}
