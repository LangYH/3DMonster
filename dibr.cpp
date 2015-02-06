#include "dibr.h"

#define VIEW_DISTANCE 0.6
#define MAX_DEPTH 0.1
#define EYES_INTERVAL 0.065
#define DISPLAYER_INTERVAL 0.000277

#include <math.h>
#include <QDebug>

DIBR::DIBR()
{
}

Mat DIBR::execute( Mat const image, Mat const depthMap)
{
    //execute the DIBR algorithm , return a Red-blue 3D image
    Mat currentDepthMap = depthMap * MAX_DEPTH + VIEW_DISTANCE;

    Mat right_movement = ( 1. - VIEW_DISTANCE / currentDepthMap ) *( EYES_INTERVAL / 2. ) / DISPLAYER_INTERVAL;
    Mat left_movement = -right_movement;
    right_movement.convertTo( right_movement, CV_8SC1 );
    left_movement.convertTo( left_movement, CV_8SC1 );

    Mat rightImage = creatRightImage( image, right_movement );
    fillImagehole( rightImage );
    Mat leftImage = creatLeftImage( image, left_movement );
    fillImagehole( leftImage );


    return creatRedBlueImage( leftImage, rightImage );
}

Mat DIBR::creatRightImage( Mat const image,Mat const right_movement )
{
    Mat rightImage( image.size(), image.type(), Scalar( 0 ) );
    for( int i = 0; i < image.rows; i++ )
    {
        //for each row
        double minValue, maxValue;
        minMaxLoc( right_movement.row( i ), &minValue, &maxValue );

        for( int currentMovement = int(minValue); currentMovement <= int(maxValue); currentMovement++ )
        {
            //start from this mininmum depth-Value to maximum depth-value
            for( int j = 0; j < image.cols; j++ )
            {
                //for each column
                if( right_movement.at<char>(i,j) == currentMovement && ( j + currentMovement ) >=  0 &&
                        ( j + currentMovement ) < image.cols && rightImage.at<Vec3b>( i, j + currentMovement)[0] == 0 )
                {
                    if( image.channels() == 1 )
                    {
                        rightImage.at<char>( i, j + currentMovement) = image.at<uchar>( i, j );
                    }
                    else
                    {
                        rightImage.at<Vec3b>( i, j + currentMovement)[0] = image.at<Vec3b>( i, j )[0];
                        rightImage.at<Vec3b>( i, j + currentMovement)[1] = image.at<Vec3b>( i, j )[1];
                        rightImage.at<Vec3b>( i, j + currentMovement)[2] = image.at<Vec3b>( i, j )[2];
                    }
                }

            }//end of column scan
        } //end of depth scan
    }//end of row scan
    return rightImage;
}

Mat DIBR::creatLeftImage(Mat const image, Mat const left_movement )
{
    Mat leftImage( image.size(), image.type(), Scalar( 0 ) );
    for( int i = 0; i < image.rows; i++ )
    {
        //for each row
        double minValue, maxValue;
        minMaxLoc( left_movement.row( i ), &minValue, &maxValue );

        for( int currentMovement = int(maxValue); currentMovement >= int(minValue); currentMovement-- )
        {
            //start from this mininmum depth-Value to maximum depth-value
            for( int j = 0; j < image.cols; j++ )
            {
                //for each column
                if( left_movement.at<char>(i,j) == currentMovement && ( j + currentMovement ) >=  0 &&
                        ( j + currentMovement ) < image.cols && leftImage.at<Vec3b>( i, j + currentMovement)[0] == 0 )
                {
                    if( image.channels() == 1 )
                    {
                        leftImage.at<char>( i, j + currentMovement) = image.at<uchar>( i, j );
                    }
                    else
                    {
                        leftImage.at<Vec3b>( i, j + currentMovement)[0] = image.at<Vec3b>( i, j )[0];
                        leftImage.at<Vec3b>( i, j + currentMovement)[1] = image.at<Vec3b>( i, j )[1];
                        leftImage.at<Vec3b>( i, j + currentMovement)[2] = image.at<Vec3b>( i, j )[2];
                    }
                }

            }//end of column scan
        } //end of depth scan
    }//end of row scan
    return leftImage;

}

Mat DIBR::creatRedBlueImage( Mat const leftImage, Mat const rightImage )
{
    Mat RedBlueImage( leftImage.size(), leftImage.type(), Scalar(0));
    Mat inputImage[] = { leftImage, rightImage };

    //leftImage[0]->RedBlueImage[0]
    //rightImage[1]->RedBlueImage[1], rightImage[2]->RedBlueImage[2]
    int from_to[] = { 2,2, 3,0, 4,1 };
    mixChannels( inputImage, 2, &RedBlueImage, 1, from_to, 3);

    return RedBlueImage;

}

void DIBR::fillImagehole(Mat &Input)
{
    for( int i = 2; i < Input.rows - 2; i++ )
    {
        for( int j = 2; j < Input.cols - 2; j++ )
        {
            if( Input.at<Vec3b>( i, j )[0] == 0 )
            {
                Mat medianRegion;
                medianRegion = Input( cv::Range( i - 2, i+2 ),
                                      cv::Range( j - 2, j+2 ) );
                medianBlur( medianRegion, medianRegion, 5 );

            }
        }

    }
}

void DIBR::depthMapNormalized( Mat &depthMap )
{
    if( !depthMap.empty() )
    {
        //normalize depthMap to [ 0, 1 ]
        double minValue, maxValue;
        minMaxLoc( depthMap, &minValue, &maxValue );

        depthMap.convertTo( depthMap, CV_64FC1 );

        depthMap = ( depthMap - minValue )/(maxValue - minValue );
    }
}
