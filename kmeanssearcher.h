#ifndef KMEANSSEARCHER_H
#define KMEANSSEARCHER_H
#include <QStringList>
#include <opencv2/opencv.hpp>
using namespace cv;

class kmeansSearcher
{
public:
    kmeansSearcher(const QStringList &input_imlist, const QStringList &input_depthlist, int k);
    ~kmeansSearcher();
    void train();
    int classify( Mat const &inputImage );
    int vq( Mat const &featureVector );

private:
    QStringList imlist;
    QStringList depthlist;

    int nbr_classes;  // k classes
    Mat centers;     // k centroid after training

    int trainImageWidth;
    int trainImageHeight;
    HOGDescriptor *hogDesr;

public:
    Mat bestlabels;  //labels for each training image
    std::vector<Mat> k_depthMaps; //k depthMaps after training
};

#endif // KMEANSSEARCHER_H
