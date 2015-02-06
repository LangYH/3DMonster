#ifndef KNNSEARCHER_H
#define KNNSEARCHER_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <QStringList>
using namespace cv;

class kNNSearcher
{
public:
    kNNSearcher();
    static void kNNSearchWithHOG(Mat const &inputImage, QStringList const &imPath,
                           Mat &indexes, Mat &weights, int k );
    static void kNNSearch(Mat const &targetDescr, Mat const &descriptorMat, Mat &indexes, Mat &weights, int k );

};

#endif // KNNSEARCHER_H
