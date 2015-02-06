#ifndef DEPTHMAPGENERATINGALGORITHMS_H
#define DEPTHMAPGENERATINGALGORITHMS_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/objdetect/objdetect.hpp>
using namespace cv;

class QString;
class QStringList;

class DepthMapGeneratingAlgorithms
{
public:
    //Algorithm1: Relative depth cue
    DepthMapGeneratingAlgorithms();
    static Mat usingRelativeHeightDepthCue( Mat const &inputImage );
    static Mat lineTracing( Mat const &edgeMap );
    static Mat createDepthMapUsingLineMap(Mat const &edgeMap, Mat const &lineMap );

    //kNN search with HOG
    static void usingkNNWithHOG(Mat const &inputImage , Mat &finalDepthMap, int k);
    static void usingkmeansWithHOG(Mat const &inputImage, Mat &finalDepthMap , int k);
};





#endif // DEPTHMAPGENERATINGALGORITHMS_H
