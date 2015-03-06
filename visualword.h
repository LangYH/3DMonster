#ifndef VISUALWORD_H
#define VISUALWORD_H
#include "opencv2/opencv.hpp"
#include <QStringList>
#include "opencv2/gpu/gpu.hpp"
using namespace cv;

enum CROSS_VALIDATION_SYMBOL { STAGE_ONE, STAGE_TWO };

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
    void visualWordsTrainingWithCrossValidation();
    bool trainOneVisualWord(CvSVM &svm, const int init_class_label , const int iteration);
    void svmTrain( CvSVM &svm, const int class_label ,CROSS_VALIDATION_SYMBOL cv_symbol );
    void svmDetect( CvSVM &svm, const int class_label, CROSS_VALIDATION_SYMBOL cv_symbol );

    //parameters setting
    void setCentroids( int k );
    void setKmeansIterations(int i);
    void setTrainingIterations(int i);
    int getDiscoverySamplesSize();

    //Data pre-processing
    static void getDataForTrainingFromDatabase(QStringList &D1, QStringList &D2, QStringList &N1, QStringList &N2);
    void computeDesriptorMat(const QStringList &D1, Mat &descriptorMatOfD1);

    //other operation
    void initDatabaseClassLabel();
    void cleanDatabaseClassLabel(CROSS_VALIDATION_SYMBOL cv_symbol );
    bool keepTopResults( const int m,
                        const int class_label, CROSS_VALIDATION_SYMBOL cv_symbol );
    void writeKmeansResultToDatabase( const Mat &bestlabels );

};

#endif // VISUALWORD_H
