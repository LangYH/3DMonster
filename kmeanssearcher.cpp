#include "kmeanssearcher.h"
#include "imtools.h"
#include <vector>
#include <QElapsedTimer>

kmeansSearcher::kmeansSearcher(QStringList const &input_imlist,
                               QStringList const &input_depthlist, int k)
{
    imlist = input_imlist;
    depthlist = input_depthlist;
    nbr_classes = k;
    hogDesr = new HOGDescriptor( cvSize( 640, 480 ), cvSize( 160, 120 ), cvSize( 160,120 ), cvSize( 160, 120 ), 9  );
}

kmeansSearcher::~kmeansSearcher()
{
    delete hogDesr;
}

void kmeansSearcher::train()
{
    QElapsedTimer timer;
    timer.start();

    //filename: this file is used to store the result
    QString const filename = "kmeansData/kmeansData_" + QString::number(nbr_classes) +".yaml";
    FileStorage fs;

    if( fs.open( filename.toLocal8Bit().data(), FileStorage::READ ) ){
        // the kmeansData.yaml does exist
        fs["height"] >> trainImageHeight;
        fs["width"]  >> trainImageWidth;
        fs["nbr_classes"] >>  nbr_classes;
        fs["bestlabels"] >>  bestlabels;
        fs["centers"] >>  centers;

        k_depthMaps.clear();

        QString matName = "Mat";
        for( int i = 0; i < nbr_classes; i++ ){
            Mat temp;
            matName = matName + QString::number(i);
            fs[matName.toLocal8Bit().data()] >>  temp;
            k_depthMaps.push_back( temp );
          //  std::cout << k_depthMaps[i] << std::endl;
        }
    }else{
        //there is no kmeansData.yaml
        //compute all descriptors of training image
        Mat descriptorsMat;
        imtools::computeHOGDescriptorsMat( descriptorsMat, imlist, hogDesr );
        //*****************************************************************************

        //deploying kmeans algorithm
        //notice:data type of the first argument of function cv::kmeans has to be CV_32FC1
        descriptorsMat.convertTo( descriptorsMat, CV_32FC1 );
        kmeans( descriptorsMat, nbr_classes, bestlabels, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0 ),
                3, KMEANS_PP_CENTERS, centers );
        //****************************************************************************

        //compute k depthmaps
        std::vector<Mat> depthMaps;
        imtools::getDepthMapsFromDepthlist( depthMaps, depthlist );

        trainImageHeight = depthMaps[0].rows;
        trainImageWidth = depthMaps[0].cols;

        for( int i = 0; i < nbr_classes; i++ ){
            Mat empty( trainImageHeight, trainImageWidth, CV_32FC1, Scalar(0) );
            k_depthMaps.push_back( empty );
        }

        std::vector<int> counts( nbr_classes, 0 );

        for( int i = 0; i < depthlist.size(); i++ ){
            int clusterIndex = bestlabels.at<int>( i, 0 );
            counts[clusterIndex] += 1;
            k_depthMaps[clusterIndex] = k_depthMaps[clusterIndex] + depthMaps[i];
        }

        //normalize and excute histequalize
        for( int i = 0; i < nbr_classes; i++ ){
            k_depthMaps[i] = k_depthMaps[i] / double( counts[i] );
            Mat im;
            imtools::matrixNormalize( k_depthMaps[i], im );
            im =  im * 255;
            im.convertTo( im, CV_8UC1 );
            equalizeHist( im, k_depthMaps[i] );
        }

        std::cout << "kmeans training finished!!!" << std::endl;
        std::cout << "time: " << QString::number( timer.elapsed() ).toDouble() / 1000.0 << " s" << std::endl;

        //after training, store in files: nbr_classes, bestlabels, centers, k_depthMaps
        if( fs.open( filename.toLocal8Bit().data(), FileStorage::WRITE ) ){
            fs << "height" << trainImageHeight;
            fs << "width" << trainImageWidth;
            fs << "nbr_classes" << nbr_classes;
            fs << "bestlabels" << bestlabels;
            fs << "centers" << centers;
            QString matName = "Mat";
            for( int i = 0; i < nbr_classes; i++ ){
                std::cout << k_depthMaps[i] << std::endl;
                matName = matName + QString::number(i);
                fs << matName.toLocal8Bit().data() << k_depthMaps[i];
            }
        }
    }
    fs.release();

}

int kmeansSearcher::classify(const Mat &inputImage)
{
    if( inputImage.empty() ){
        return -1;
    }
    Mat im;
    if( inputImage.size() != Size( trainImageWidth, trainImageHeight ) ){
        resize( inputImage, im, Size( trainImageWidth, trainImageHeight ) );
    }else{
        inputImage.copyTo( im );
    }
    if( im.channels() == 3 ){
        im.convertTo( im, CV_BGR2GRAY);
        im.convertTo( im, CV_8UC1 );
    }

    std::vector<float> targetDescriptor;
    hogDesr->compute( im, targetDescriptor, Size(0,0), Size(0, 0) );

    Mat targetDescr( targetDescriptor );

    return vq( targetDescr.t() );

}

int kmeansSearcher::vq( Mat const &featureVector )
{
    Mat centroids = centers.clone();
    int nr = centroids.rows;
    int nc = centroids.cols;
    //for each row of descriptorMat, substract by targetDescr and square
    //( a - b )^2
    Mat feature;
    featureVector.convertTo( feature, CV_32FC1 );
    centroids.convertTo( centroids, CV_32FC1 );
    Mat diffMat( nr, nc, CV_32FC1 );
    for( int i = 0; i < nr; i++ ){
        const double *data_feature = feature.ptr<double>(0);
        double *data_centroids = centroids.ptr<double>(i);
        double *data_diffMat = diffMat.ptr<double>(i);
        for( int j = 0; j < nc; j++ ){
            double temp = (*data_centroids++)-(*data_feature++);
            *data_diffMat++ = temp * temp;
        }
    }

    //squareroot of summation of each row: ( a11 + a12 + a13 ..)^1/2
    Mat sumAndSqrtVector( nr, 1, CV_32FC1 );
    for( int i = 0; i < nr; i++ ){
        std::cout << sum( diffMat.row(i) )[0]  << std::endl;
        sumAndSqrtVector.at<double>( i, 0 ) = std::sqrt( sum( diffMat.row(i) )[0] );
    }

    Mat ides;
    sortIdx( sumAndSqrtVector, ides, CV_SORT_ASCENDING + CV_SORT_EVERY_COLUMN );

    return ides.at<ushort>( 0, 0 );


}

