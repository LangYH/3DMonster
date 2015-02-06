#ifndef VISUALWORD_H
#define VISUALWORD_H
#include "opencv2/opencv.hpp"
#include <QStringList>
using namespace cv;

class VisualWord
{
public:
    VisualWord();
    void train();
    void getDataForTraining(QStringList &D1, QStringList &D2, QStringList &N1, QStringList &N2);
    void getDiscoverySamples( const QStringList &D1, QStringList &samplesOfD1);
    void computeDesriptorMat(const QStringList &samplesOfD1, Mat &descriptorMatOfD1);
    void getNegativeSamples( const QStringList &N1, QStringList &samplesOfN1);
};

#endif // VISUALWORD_H
