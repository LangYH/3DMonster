#ifndef LINEEXTRACTINGALGORITHMS_H
#define LINEEXTRACTINGALGORITHMS_H


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

class LineExtractingAlgorithms
{
public:
    LineExtractingAlgorithms();
    static Mat canny(Mat const &inputImage );

private:
    static void get_coor( int &x1, int &y1, int &x2, int &y2, double orc );

};

#endif // LINEEXTRACTINGALGORITHMS_H
