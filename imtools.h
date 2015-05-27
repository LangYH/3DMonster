#ifndef IMTOOLS_H
#define IMTOOLS_H
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <QDir>

using namespace cv;

typedef double ElementType;


class imtools
{
public:
    imtools();

    //normalize a matrix to [0,1]
    static void matrixNormalize(const Mat &srcMat, Mat &dstMat );
    static void weightsNormalize( Mat &matrix );//inplace version

    //giving the image and depthmap path, return the imlist and depthlist
    static void getImListAndDepthList( QString const &imPath, QString const &depthPath,
                                QStringList &imlist, QStringList &depthlist );

    //get the depthmaps respect to the given indexes
    static void getDepthMapsWithIndexes(const Mat &indexes,
                                        const QStringList &depthlist, std::vector<Mat> &depthMaps );

    //fuse the depthmaps in depthMaps to a single depthmap according to weights
    static void fuseDepthMaps(const std::vector<Mat> &depthMaps,
                              const Mat &weights, Mat &fusedDepthMap );

    //compute all the HOG descriptor in imPath, each row of descriptorMat is a descriptor of one image
    static void computeHOGDescriptorsMat(Mat &descriptorMat,
                                         const QStringList &imPath, const HOGDescriptor *hogDesr);

    //get all the depthmap from depthlist, store in depthMaps
    static void getDepthMapsFromDepthlist(vector<Mat> &depthMaps,
                                          QStringList const &depthlist );

    //compute the gradient energy of the input matrix
    //for a smooth area, it give a small value
    static double computeGradientEnergyWithHOG(const Mat &patch);

    static void idxSort(const std::vector<double> data, std::vector<int> &sorted_index , bool reverse = false );
    static void idxSort(ElementType Data[], int SortedIndex[], int N);
};

#endif // IMTOOLS_H
