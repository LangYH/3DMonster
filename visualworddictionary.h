#ifndef VISUALWORDDICTIONARY_H
#define VISUALWORDDICTIONARY_H
#include "opencv2/opencv.hpp"
#include "visualword.h"
#include "CustomType.h"
using namespace cv;

class VisualWordDictionary
{
private:
    HOGDescriptor *hog_descr;
    std::map<int, MySVM*> classifiers;
    //std::vector<Mat> natural_images;
    QStringList natural_images_list;

public:
    VisualWordDictionary();
    ~VisualWordDictionary();

    bool trainDictionary();

    bool loadDictionary();

    void prepareNegativeSamplesForMultipleSVMTraining( Mat &negativeDescriptorMat );

    bool trainMultipleSVMClassifier();

    void getAllPositiveSample(std::map<int, Mat> &positive_samples, double score_threshold);

    void getHardExmaples(MySVM &svm, std::vector<Mat> &hard_examples);

    bool loadAllSVMClassifiers();

    void cleanAllSVMClassifiers();

    //void searchImageForGivenWord( const Mat &image,
    //                              MySVM &svm,
    //                              std::vector<Rect> rect_founded );


    static void getSVMDetectorForHOG( MySVM *svm, std::vector<float> &myDetector );

    int searchForId(const Mat &patch , double &score);
    Mat searchForDepthWithGiveID(int id);
    static void searchImageForGivenWord(const Mat &image, MySVM &svm,
                                 std::vector<Rect> &filtered_result,
                                 std::vector<double> &filtered_scores);

    void loadNaturalImages();

    void getCanonicalPatchesForGiveRects(const Mat &image, const std::vector<Rect> founded_rect, std::vector<Mat> &patches);

};

#endif // VISUALWORDDICTIONARY_H
