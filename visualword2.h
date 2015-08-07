#ifndef VISUALWORD2_H
#define VISUALWORD2_H
#include "opencv2/opencv.hpp"
#include <QStringList>
#include "opencv2/gpu/gpu.hpp"
#include <CustomType.h>
using namespace cv;

class PatchInfo;

class VisualWord2
{
private:
    QStringList I1, I2, D1, D2, N1, N2;
    Mat negative_descriptor_1, negative_descriptor_2;
    std::vector<int> init_class_label;
    std::vector<int> visual_word_IDs;
    std::map<int, QStringList> visual_word_patches;
    std::map<int,QString> visual_word_paths;

    std::map<int, double> cluster_svm_scores;
    std::map<int, QStringList> clusters;
    HOGDescriptor *hog_descr;

public:
    VisualWord2();
    ~VisualWord2();
    void loadDataFromDatabase();
    void kmeansInitialization();
    void detectTopPatches(MySVM &svm, const QStringList &target_depth_list, const QStringList &target_image_list,
                          std::vector<PatchInfo> &depth_patches,
                          int m);
    static void getHardExmaples(MySVM &svm, const QStringList &negative_images_list, std::vector<Mat> &hard_examples);
    void train();
    bool trainOneSingleVisualWord(int class_label, int iterations, MySVM &svm);
    bool isEqualPatchesCluster(std::vector<PatchInfo> &found_depth_patches, std::vector<PatchInfo> &detect_depth_patches);
    static void trainOneSVMDetectorWithHardExamples(const Mat &positive_descriptor_Mat, const Mat &negative_descriptor_Mat, const std::vector<Mat> &hard_examples, HOGDescriptor *hog_descr, MySVM &svm);
    void getPositiveSamplesForGivenID(int class_label, std::vector<PatchInfo> &detect_depth_patches_D1);
    void trainOneSVMDetector(const std::vector<PatchInfo> &detect_depth_patches, const QStringList &negative_image_list, CROSS_VALIDATION_SYMBOL cv_symbol, MySVM &svm);
    static void svmTrain(const Mat &positive_descriptor_Mat, const Mat &negative_descriptor_Mat, MySVM &svm);
    void getNegativeDescriptorForGivenStage(CROSS_VALIDATION_SYMBOL cv_symbol, Mat &negative_descriptor_Mat);
    void computeHOGDescriptorsMat(const std::vector<PatchInfo> &detect_depth_patches, Mat &descriptorMat, const HOGDescriptor *hogDesr);
    void storeClusters(int target_class_label, const std::vector<PatchInfo> &detect_patches_D1, const std::vector<PatchInfo> &detect_patches_D2);
    void parallelTrain();
    void updateDataBase();

    //train visual word detector
    void trainVisualWordDetector(int class_ID, int iteration, MySVM &svm);
    void parallelTrainVisualWordDetector();
    void trainVisualWordInitialization();
};

#endif // VISUALWORD2_H
