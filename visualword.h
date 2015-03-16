#ifndef VISUALWORD_H
#define VISUALWORD_H
#include "opencv2/opencv.hpp"
#include <QStringList>
#include "opencv2/gpu/gpu.hpp"
using namespace cv;

enum CROSS_VALIDATION_SYMBOL { STAGE_ONE, STAGE_TWO };
//notice:
//At STAGE_ONE: D1, N1 as training data, D2 as discovery data
//At STAGE_TWO: D2, N2 as trianing data, D1 as discovery data

class VisualWord
{
private:
    //paramaters
    int centroids;
    int kmeans_iterations;
    int training_iterations;
    int m;

    HOGDescriptor *hog_descr;
    CvSVMParams svm_params;

    //training data
    QStringList D1, D2, N1, N2;

    std::vector<int> D1_label;
    std::vector<float> D1_score;
    std::vector<int> D2_label;
    std::vector<float> D2_score;

    //kmeans initialization result
    Mat centers;
    Mat bestlabels;


public:
    VisualWord();
    ~VisualWord();

    bool data_loaded;
    bool isDataLoaded();

    //algorithm steps
    void train();
    void loadDataFromDatabase();
    void kmeansInitialize();

    //train one visual word
    bool trainOneVisualWord(CvSVM &svm, const int init_class_label , const int iteration);

    void svmTrain( CvSVM &svm, const int class_label , CROSS_VALIDATION_SYMBOL cv_symbol );
    int svmDetect( CvSVM &svm, const int class_label, CROSS_VALIDATION_SYMBOL cv_symbol );

    void initDatabaseClassLabel();
    void updateDatabase();
    void cleanClassLabel( CROSS_VALIDATION_SYMBOL cv_symbol );
    void getPositiveSampleList( QStringList &positive_list, const int class_label,
                                CROSS_VALIDATION_SYMBOL cv_symbol );
    void keepTopResults( const int m, const int class_label, CROSS_VALIDATION_SYMBOL cv_symbol );

    //**************************************************************************************

    //parameters setting
    void setCentroids( int k );
    void setKmeansIterations(int i);
    void setTrainingIterations(int i);
    int getDiscoverySamplesSize();

    //Data pre-processing
    static void getDataForTrainingFromDatabase(QStringList &D1, QStringList &D2, QStringList &N1, QStringList &N2);
    void computeDesriptorMat(const QStringList &D1, Mat &descriptorMatOfD1);

    //other operation
    void writeKmeansResultToDatabase( const Mat &bestlabels );

};

#endif // VISUALWORD_H
