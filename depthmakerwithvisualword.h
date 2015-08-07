#ifndef DEPTHMAKERWITHVISUALWORD_H
#define DEPTHMAKERWITHVISUALWORD_H
#include <opencv2/core/core.hpp>
#include "visualworddictionary.h"
using namespace cv;


class DepthMakerWithVisualWord
{
private:
    VisualWordDictionary *dict;

public:
    DepthMakerWithVisualWord();
    ~DepthMakerWithVisualWord();
    Mat generateDepthmapInMultiScale( const Mat image );
    void searchImageForGivenWord(const Mat &image, MySVM &svm,
                                  std::vector<Rect> &filtered_result,
                                  std::vector<double> &filtered_scores );

    void generateDepthmap(const Mat &image , std::vector<Mat> &results);
    void generateInitDepthMap( std::vector<Mat> const &images, std::vector<Mat> &depths );
    void initialDepthMap( std::vector<Mat> const &images, std::vector<Mat> &depths );
    void samplePatchesForDepthGeneration( const Mat &image,
                                          std::vector<Mat> &pyrs,
                                          std::vector< std::vector<Mat> > &patches,
                                          std::vector< std::vector<Point> > &coordinates );

    void sortedClassifiedResults( std::vector< std::vector< double > > score_array,
                            std::vector< std::vector< int > > &sorted_index );

    void classifyAllPatches( const std::vector< std::vector< Mat > > patches_array,
                             std::vector< std::vector< int > > &result_class,
                             std::vector< std::vector< double > > &svm_score );
};

#endif // DEPTHMAKERWITHVISUALWORD_H
