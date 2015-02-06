#include "visualword.h"

VisualWord::VisualWord()
{
}

void VisualWord::train()
{
    //---------Step1: Initailization( mainly use kmeans )-----------------------------
    QStringList D1, D2, N1, N2;
    getDataForTraining( D1, D2, N1, N2 );

    QStringList samplesOfD1;
    getDiscoverySamples( D1, samplesOfD1 );

    Mat descriptorMatOfD1;
    computeDesriptorMat( samplesOfD1, descriptorMatOfD1 );

    int k = 3;
    Mat centers, bestlabels;
    kmeans( descriptorMatOfD1, k, bestlabels, TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0 ), 3,
            KMEANS_PP_CENTERS, centers );

    //----------Step2:Iteration process ( mainly use SVM and cross validation )--------------------
    QStringList samplesOfN1;
    getNegativeSamples( N1, samplesOfN1 );




}


 void VisualWord::getDataForTraining( QStringList &D1, QStringList &D2,
                                      QStringList &N1, QStringList &N2 )
 {

 }

void VisualWord::getDiscoverySamples( const QStringList &D1, QStringList &samplesOfD1 )
{

}

void VisualWord::computeDesriptorMat(const QStringList &samplesOfD1, Mat &descriptorMatOfD1 )
{

}

void VisualWord::getNegativeSamples(const QStringList &N1, QStringList &samplesOfN1 )
{

}
