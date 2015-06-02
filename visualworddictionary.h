#ifndef VISUALWORDDICTIONARY_H
#define VISUALWORDDICTIONARY_H
#include "opencv2/opencv.hpp"
#include "visualword.h"
using namespace cv;

class MySVM : public CvSVM
{
public:
    double * get_alpha_vector()
    {
        return this->decision_func->alpha;
    }

    float get_rho()
    {
        return this->decision_func->rho;
    }

    std::vector<float> get_primal_form() const
    {
      std::vector<float> support_vector;

      int sv_count = get_support_vector_count();

      const CvSVMDecisionFunc* df = decision_func;
      const double* alphas = df[0].alpha;
      double rho = df[0].rho;
      int var_count = get_var_count();

      support_vector.resize(var_count, 0);

      for (unsigned int r = 0; r < (unsigned)sv_count; r++)
      {
        float myalpha = alphas[r];
        const float* v = get_support_vector(r);
        for (int j = 0; j < var_count; j++,v++)
        {
          support_vector[j] += (-myalpha) * (*v);
        }
      }

      support_vector.push_back(rho);

      return support_vector;
    }

};

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
