#ifndef DIBR_H
#define DIBR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

class DIBR
{
public:
    DIBR();

public:
    //method
    Mat creatRightImage(const Mat image, Mat const right_movement );
    Mat creatLeftImage(const Mat image, Mat const left_movement );
    Mat creatRedBlueImage( Mat  const leftImage, Mat const rightImage );
    Mat execute( Mat const image, Mat const depthMap);
    void fillImagehole(Mat &Input );
    void depthMapNormalized( Mat &depthMap );

};

#endif // DIBR_H
