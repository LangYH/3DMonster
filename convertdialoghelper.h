#ifndef CONVERTDIALOGHELPER_H
#define CONVERTDIALOGHELPER_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "dibr.h"
using namespace cv;

class ConvertDialogHelper
{
public:
    ConvertDialogHelper();
    ~ConvertDialogHelper();
    DIBR *DIBRAlgorithm;

private:
    Mat image;
    Mat depthImage;
    Mat depthImageToDisplay; //proper depth image for displaying on widget;
    Mat result;


public:
    bool setInputImage( Mat input );
    bool setDepthImage( Mat depth );
    const Mat getLastResult() const;
    const Mat getLastDepthMapResult();
    void process();
    void generateDepthMapUsingRelativeHeightCue();

};

#endif // CONVERTDIALOGHELPER_H
