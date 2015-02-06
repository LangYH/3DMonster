#ifndef STATISTIC_H
#define STATISTIC_H
#include "opencv2/core/core.hpp"
using namespace cv;

class Statistic
{
public:
    Statistic();

    //compute the SSIM between two matrix
    static Scalar computeSSIMGaussian(const Mat &matrix1, const Mat &matrix2);
    static double computeSSIM(const Mat &matrix1, const Mat &matrix2 );

    //compute the mean value and deviation
    static void computeMeanAndDeviation(const Mat &patch, double &mean_value, double &standard_deviation );

    //compute cross deviation of two matrix
    static double computeCrossDeviation(const Mat &matrix1, const Mat &matrix2);

    //compute mean value
    static double computeMeanValue(const Mat &patch );



    //compute cross correlation between two matrix
    //represented by angle between 0~180 degree, small value means the two patches is very relavent
    //to each other
    static double computeCrossCorrelation(const Mat &patch1, const Mat &patch2);

    //compute cosine distance of two matrix
    static double computeCosineDistance(const Mat &patch1, const Mat &patch2);

};

#endif // STATISTIC_H
