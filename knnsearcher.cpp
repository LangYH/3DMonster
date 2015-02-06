#include "knnsearcher.h"
#include <iostream>
#include "imtools.h"

kNNSearcher::kNNSearcher()
{
}

void kNNSearcher::kNNSearchWithHOG(const Mat &inputImage, const QStringList &imPath,
                      Mat &indexes, Mat &weights, int k)
{
    //resize inputImage to the same size of training image
    Mat temp = imread( imPath[0].toLocal8Bit().data(), CV_LOAD_IMAGE_GRAYSCALE );
    Mat inputIm;
    resize( inputImage, inputIm, temp.size() );

    //compute the HOG descriptor of target image
    HOGDescriptor *hogDesr = new HOGDescriptor( cvSize( 640, 480 ), cvSize( 160, 120 ), cvSize( 160,120 ), cvSize( 160, 120 ), 9  );
    std::vector<float> targetDescriptor;
    hogDesr->compute( inputIm, targetDescriptor, Size( 0, 0 ), Size( 0, 0) );
    //###################################################################################

    //load the training descriptors into descriptorMat if there exist a HOGof44blocks.yaml file
    //otherwise, execute the train program
    Mat descriptorMat;

    QString const HOGMatfile = "HOGof44blocks.yaml";
    FileStorage fs;
    fs.open( HOGMatfile.toLocal8Bit().data(), FileStorage::READ );
    if( fs.isOpened() ){
        // the HOGof44blocks.yaml does exist
        fs["HOGMat"] >> descriptorMat;
    }else{
        std::cout<<"HOGMat has not been created, now computing..." << std::endl;
        fs.release();
        //if the file "HOGof44blocks.yaml" is not created,
        imtools::computeHOGDescriptorsMat( descriptorMat, imPath, hogDesr );

        //storge the HOGMat
        fs.open( HOGMatfile.toLocal8Bit().data(), FileStorage::WRITE );
        fs << "HOGMat" << descriptorMat;
        fs.release();
    }
    //##########################################################################################

    Mat targetDescr( targetDescriptor );
    targetDescr.convertTo( targetDescr, CV_64FC1 );

    kNNSearch( targetDescr.t(), descriptorMat, indexes, weights, k );

}


void kNNSearcher::kNNSearch(const Mat &targetDescr, const Mat &descriptorMat, Mat &indexes, Mat &weights, int k )
{
    if( targetDescr.empty() || descriptorMat.empty() ){
        std::cout << "Descriptors is empty!" << std::endl;
        return;
    }

    int nr = descriptorMat.rows;
    int nc = descriptorMat.cols;
    //for each row of descriptorMat, substract by targetDescr
    Mat titlMat( nr, nc, descriptorMat.type() );
    for( int i = 0; i < nr; i++ ){
        const double *data_t = targetDescr.ptr<double>(0);
        double *data = titlMat.ptr<double>(i);
        for( int j = 0; j < nc; j++ ){
            *data++ = *data_t++;
        }
    }

    Mat diffMat;
    diffMat = descriptorMat - titlMat;

    diffMat = diffMat.mul( diffMat );

    Mat sumAndSqrtVector( nr, 1, CV_64FC1 );
    for( int i = 0; i < nr; i++ ){
        sumAndSqrtVector.at<double>( i, 0 ) = std::sqrt( sum( diffMat.row(i) )[0] );
    }

    Mat ides;
    sortIdx( sumAndSqrtVector, ides, CV_SORT_ASCENDING + CV_SORT_EVERY_COLUMN );

    sort( sumAndSqrtVector, sumAndSqrtVector, CV_SORT_ASCENDING+CV_SORT_EVERY_COLUMN );

    indexes = ides.rowRange( Range( 0, k ) );
    weights = sumAndSqrtVector.rowRange( Range( 0, k ) );
    weights = 1.0 / weights;
    imtools::weightsNormalize( weights );
}


