#ifndef FILTERS_H
#define FILTERS_H
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;

class Filters
{
public:
    Filters();
    static void crossBilateralFilter(Mat const &srcImage, Mat const &srcmask,
                               Mat &dstImage, int wsize, double sigma_space, double sigma_value );
    static void guidedFilter( Mat const &srcImage, Mat const &mask, Mat &dstImage, int wsize, double regularzationTerm );

};

#endif // FILTERS_H
