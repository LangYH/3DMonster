#ifndef VISUALWORDDICTIONARY_H
#define VISUALWORDDICTIONARY_H
#include "opencv2/opencv.hpp"
#include "visualword.h"
using namespace cv;

class VisualWordDictionary : public VisualWord
{
private:
    HOGDescriptor *hog_descr;
    std::map<int, CvSVM*> classifiers;

public:
    VisualWordDictionary();

    bool trainDictionary();

    bool loadDictionary();

    void prepareNegativeSamplesForMultipleSVMTraining( Mat &negativeDescriptorMat );

    bool trainMultipleSVMClassifier();

    void cleanAllSVMClassifiers( std::map<int, CvSVM*> &classifiers );

    void loadAllSVMClassifiers( std::map<int, CvSVM*> &classifiers );

    Mat searchForDepth( const Mat &patch );
    Mat searchForId( const Mat &patch );
};

#endif // VISUALWORDDICTIONARY_H
