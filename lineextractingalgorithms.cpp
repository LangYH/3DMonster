#include "lineextractingalgorithms.h"
#include <math.h>
using namespace cv;

LineExtractingAlgorithms::LineExtractingAlgorithms()
{
}

Mat LineExtractingAlgorithms::canny(const Mat &inputImage)
{
    if( !inputImage.empty() )
    {
        Mat image;
        inputImage.copyTo( image );

        //GaussianBlue kernel:3*3
        //cv::GaussianBlur( image, image, cv::Size( 5, 5 ), 0, 0, BORDER_DEFAULT );

        //convert image to gray-level
        Mat grayImage;

        cvtColor( image, grayImage, COLOR_BGR2GRAY );


        //generate gradient along x and y
        Mat grad_x, grad_y;

        //x direction
        cv::Sobel( grayImage, grad_x, CV_64F, 1, 0 );

        //y direction
        cv::Sobel( grayImage, grad_y, CV_64F, 0, 1 );

        //total gradient
        // grad = sqrt( grad_x ^ 2 + grad_y ^ 2 )
        Mat grad_x_square, grad_y_square, grad;
        cv::pow( grad_x, 2, grad_x_square );
        cv::pow( grad_y, 2, grad_y_square );
        cv::sqrt( grad_x_square + grad_y_square, grad );

        //非极大抑制
        Mat new_edge( grad.size(), grad.type(), Scalar(0) );
        for( int i = 1; i < grad.rows - 1; i++ )
        {
            for( int j = 1; j < grad.cols - 1; j++ )
            {
                double Mx = grad_x.at<double>( i, j );
                double My = grad_y.at<double>( i, j );
                double orc = 0.0;
                if( My != 0.0 )
                {
                    orc = std::atan2( My, Mx );
                }
                else if( My == 0.0 && Mx > 0 )
                {
                    orc = CV_PI / 2;
                }
                else
                {
                    orc = -CV_PI / 2;
                }
                //法线两侧的像素点坐标，插值需要
                int x1, y1, x2, y2 ;
                get_coor( x1, y1, x2, y2, orc );
                //插值后得到的像素
                double M1 = My * grad.at<double>( i + y1, j + x1 ) + (Mx - My)*grad.at<double>( i + y2, j + x2 );
                //法线另一侧的两点坐标，和插值后得到的像素
                get_coor( x1, y1, x2, y2, orc + CV_PI );
                double M2 = My * grad.at<double>( i + y1, j + x1 ) + (Mx - My)*grad.at<double>( i + y2, j + x2 );

                //只取比两边点都大,或都小（负值）的值
                if( (Mx*grad.at<double>(i,j)>M1)*(Mx*grad.at<double>(i,j)>=M2)+(Mx*grad.at<double>(i,j)<M1)*(Mx*grad.at<double>(i,j)<=M2) )
                {
                    new_edge.at<double>( i, j ) = grad.at<double>( i,j );
                }
            }
        }
        return( new_edge );
    }
    else{
        return Mat();
    }
}

void LineExtractingAlgorithms::get_coor( int &x1, int &y1, int &x2, int &y2, double orc )
{
    //取得对应orc弧度直线的两侧坐标
    double sigma=0.000000001;
    x1 = cvCeil( std::cos( orc + CV_PI / 8 ) * std::sqrt( 2.0 ) - 0.5 - sigma );
    y1 = cvCeil( -std::sin( orc - CV_PI / 8 ) * std::sqrt( 2.0 ) - 0.5 - sigma );
    x2 = cvCeil( std::cos( orc - CV_PI / 8 ) * std::sqrt( 2.0 ) - 0.5 - sigma );
    y2 = cvCeil( -std::sin( orc - CV_PI / 8 ) * std::sqrt( 2.0 ) - 0.5 - sigma );
}
