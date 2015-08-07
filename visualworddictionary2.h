#ifndef VISUALWORDDICTIONARY2_H
#define VISUALWORDDICTIONARY2_H
#include "opencv2/opencv.hpp"
#include <QStringList>
#include "opencv2/gpu/gpu.hpp"
#include <CustomType.h>
using namespace cv;

class MySVM;

class VisualWordDictionary2
{
public:
    VisualWordDictionary2();

    void parallelTrainVisualWordDetector();
    void getNegativeDescriptorTraining(Mat &negative_descriptor_Mat);
    void computePersonSimMatrix(std::vector<int> depths_IDs, std::map<int, Mat> depths, Mat &person_sim_matrix);
    void cleanOverlapClusters(double threshold = 0.95 );
private:
    QStringList I1, I2, D1, D2, N1, N2;

    HOGDescriptor *hog_descr;

    Mat negative_descriptor_1, negative_descriptor_2;

    std::vector<int> visual_word_IDs;
    std::map<int, QStringList> visual_word_patches;
    void trainVisualWordInitialization( bool clean_overlaps = false,
                                        double threshold = 0.95 );
    void trainVisualWordDetector(int class_ID, int iteration, MySVM &svm);
    void getNegativeDescriptorForGivenStage(CROSS_VALIDATION_SYMBOL cv_symbol, Mat &negative_descriptor_Mat);
    void trainOneSVMDetectorWithHardExamples(const Mat &positive_descriptor_Mat, const Mat &negative_descriptor_Mat, const std::vector<Mat> &hard_examples, MySVM &svm);
};

#endif // VISUALWORDDICTIONARY2_H
