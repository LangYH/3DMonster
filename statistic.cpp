#include "statistic.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>

Statistic::Statistic()
{
}

Scalar Statistic::computeSSIMGaussian(const Mat &matrix1, const Mat &matrix2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;

    Mat I1, I2;
    matrix1.convertTo(I1, d);           // cannot calculate on one byte large values
    matrix2.convertTo(I2, d);

    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
    return mssim;
//    double C1 = 6.5025, C2 = 58.5225;
//    Mat A, B;
//    matrix1.convertTo( A, CV_32F );
//    matrix2.convertTo( B, CV_32F );

//    Mat A_2 = A.mul( A ); // A^2
//    Mat B_2 = B.mul( B ); // B^2
//    Mat A_B = A.mul( B ); // A*B

//    Mat blurA, blurB;
//    GaussianBlur( A, blurA, Size( 11, 11 ), 1.5 );
//    GaussianBlur( B, blurB, Size( 11, 11 ), 1.5 );

//    Mat blurA_2 = blurA.mul( blurA );
//    Mat blurB_2 = blurB.mul( blurB );
//    Mat blurA_B = blurA.mul( blurB );

//    //E( (A-E(A))^2 ] = E( A^2 ) - E(A)^2
//    Mat sigmaA_2, sigmaB_2, sigmaA_B;
//    GaussianBlur( A_2, sigmaA_2, Size( 11, 11 ), 1.5 );
//    sigmaA_2 -= blurA_2;
//    GaussianBlur( B_2, sigmaB_2, Size( 11, 11 ), 1.5 );
//    sigmaB_2 -= blurB_2;
//    GaussianBlur( A_B, sigmaA_B, Size( 11, 11 ), 1.5 );
//    sigmaA_B -= blurA_B;

//    //SSIM = ( 2uXuY + C1 )( 2sigmaXY + C2 ) / ( uX^2 + uY^2 + C1 )( sigmaX^2 + sigmaY^2 + C2 )
//    Mat t1 = 2 * blurA_B + C1;
//    Mat t2 = 2 * sigmaA_B + C2;
//    Mat term1 = t1.mul( t2 );
//    t1 = blurA_2 + blurB_2 + C1;
//    t2 = sigmaA_2 + sigmaB_2 + C2;
//    Mat term2 = t1.mul( t2 );
//    Mat ssim_map;
//    divide( term1, term2, ssim_map );

//    return computeMeanValue( ssim_map );
}

double Statistic::computeSSIM(const Mat &matrix1, const Mat &matrix2)
{
    double L = std::pow( 2, matrix1.elemSize1() ) - 1;
    double C1 = std::pow( 0.01 * L, 2 ), C2 = std::pow( 0.03 * L, 2 );
    double mean_value1, mean_value2, deviation1, deviation2, cross_deviation;

    computeMeanAndDeviation( matrix1, mean_value1, deviation1 );
    computeMeanAndDeviation( matrix2, mean_value2, deviation2 );

    cross_deviation = computeCrossDeviation( matrix1, matrix2 );

    return ( 2 * mean_value1 * mean_value2 + C1 ) * ( 2 * cross_deviation + C2 ) /
            ( std::pow( mean_value1, 2 ) + std::pow( mean_value2, 2 ) + C1 ) /
            ( std::pow( deviation1, 2 ) + std::pow( deviation2, 2 ) + C2 );
}

void Statistic::computeMeanAndDeviation( const Mat &patch, double &mean_value, double &standard_deviation)
{
    double nbr_pixels = patch.rows * patch.cols * patch.channels();
    mean_value = computeMeanValue( patch );

    double sum_of_square = 0.;

    int nr = patch.rows;
    int nc = patch.cols * patch.channels();
    for( int i = 0; i < nr; i++){
        const uchar* data = patch.ptr<uchar>(i);
        for( int j = 0; j < nc; j++ ){
            sum_of_square += std::pow( (*data++ - mean_value ), 2. );
        }
    }

    standard_deviation = std::sqrt( (1./nbr_pixels) * sum_of_square );
}

double Statistic::computeCrossDeviation( const Mat &matrix1, const Mat &matrix2 )
{
    CV_Assert( matrix1.rows == matrix2.rows && matrix1.cols == matrix2.cols );
    double mean_value1 = computeMeanValue( matrix1 );
    double mean_value2 = computeMeanValue( matrix2 );

    double sum_of_square = 0;
    int nr = matrix1.rows;
    int nc = matrix1.cols * matrix1.channels();
    for( int i = 0; i < nr; i++ ){
        const uchar* data1 = matrix1.ptr<uchar>(i);
        const uchar* data2 = matrix2.ptr<uchar>(i);
        for( int j = 0; j < nc; j++ ){
            sum_of_square += (*data1++ - mean_value1) * (*data2++ - mean_value2 );
        }
    }

    return std::sqrt( (1./(matrix1.total()*matrix1.channels())) * sum_of_square );

}

double Statistic::computeMeanValue( const Mat &patch ){
    int nbr_pixels = patch.total() * patch.channels();

    double sum_of_pixels = 0.;

    int nr = patch.rows;
    int nc = patch.cols * patch.channels();
    for( int i = 0; i < nr; i++){
        const uchar* data = patch.ptr<uchar>(i);
        for( int j = 0; j < nc; j++ ){
            sum_of_pixels += *data++;
        }
    }

    return sum_of_pixels / double(nbr_pixels);

}

double Statistic::computeCrossCorrelation(const Mat &patch1, const Mat &patch2)
{
    CV_Assert( patch1.rows == patch2.rows && patch1.cols == patch2.cols );

    double mean_value1, mean_value2, standard_deviation1, standard_deviation2;
    Statistic::computeMeanAndDeviation( patch1, mean_value1, standard_deviation1 );
    Statistic::computeMeanAndDeviation( patch2, mean_value2, standard_deviation2 );

    int nr = patch1.rows;
    int nc = patch1.cols * patch1.channels();
    double sum_of_pixel_product = 0.;
    for( int i = 0; i < nr; i++ ){
        const uchar* data1 = patch1.ptr<uchar>(i);
        const uchar* data2 = patch2.ptr<uchar>(i);
        for( int j = 0; j < nc; j++ ){
            sum_of_pixel_product += (*data1++) * (*data2++);
        }
    }

    int nbr_pixels = patch1.total() * patch1.channels();

    double cross_correlation_score =
            ( ( 1./nbr_pixels ) * sum_of_pixel_product - mean_value1*mean_value2 )
            / (standard_deviation1*standard_deviation2 );

    double cross_correlation_score_angle = std::acos( cross_correlation_score ) * (180./CV_PI);

    return cross_correlation_score_angle;
}

double Statistic::computeCosineDistance( const Mat &patch1, const Mat &patch2 )
{
    CV_Assert( patch1.rows == patch2.rows && patch1.cols == patch2.cols );

    Mat A, B;
    if( patch1.channels() == 3 || patch2.channels() == 3 ){
        cvtColor( patch1, A, CV_BGR2GRAY );
        cvtColor( patch2, B, CV_BGR2GRAY );
    } else {
        A = patch1.clone();
        B = patch2.clone();
    }

    double sum_A_B = 0; // sum of per pixel mulplication between patch1 and patch1
    double sum_A_2 = 0; // sum of per pixel square of patch1
    double sum_B_2 = 0; // sum of per pixel square of patch2

    int nr = A.rows;
    int nc = A.cols * A.channels();
    for( int i = 0; i < nr; i++ ){
        const uchar* data1 = A.ptr<uchar>(i);
        const uchar* data2 = B.ptr<uchar>(i);
        for( int j = 0; j < nc; j++ ){
            sum_A_B += (*data1) * (*data2);
            sum_A_2 += std::pow( *data1++, 2.0 );
            sum_B_2 += std::pow( *data2++, 2.0 );
        }
    }

    double cos_value = sum_A_B / ( std::sqrt(sum_A_2) * std::sqrt(sum_B_2) );
    double cos_angle = std::acos( cos_value ) * ( 180 / CV_PI );

    return cos_angle;

}

double Statistic::computeDeviationOfMatrixes(const std::vector<Mat> &patches)
{
    if( patches.size() == 0 )
        return -1.0;

    std::vector<Mat> descrs;
    for( unsigned i = 0; i < patches.size(); i++ ){
        Mat temp;
        equalizeHist( patches[i], temp );
        descrs.push_back( patches[i] );
    }

    //compute mean vector
    Mat mean_vector = descrs[0].clone();
    for( unsigned i = 1; i < descrs.size(); i++ ){
        mean_vector += descrs[i];
    }
    mean_vector /= double( descrs.size() );

    //compute deviation
    double sigma_2 = 0.0;
    for( unsigned i = 0; i < descrs.size(); i++ ){
        Mat t1, t2;
        subtract( descrs[i], mean_vector, t1 );
        multiply( t1, t1, t2 );
        sigma_2 += cv::sum( t2 )[0];
    }

    double dev = std::sqrt( sigma_2 / descrs.size() );

    return dev;
}
