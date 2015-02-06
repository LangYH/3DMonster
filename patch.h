#ifndef PATCH_H
#define PATCH_H
#include "parameters.h"
#include "opencv2/core/core.hpp"
#include <QString>
using namespace cv;

enum PATCH_TYPE { POSITIVE, OVERLAP, FLAT };
enum OVERLAP_METRIC { SSIM, CROSS_CORRELATION };
enum FLAT_METRIC { DEVIATION, HOG_ENERGY };

class Patch
{
public:
    Patch( int patchSize );
    ~Patch();

    //sample patches in pyramids:version 1
    void randomSamplePatchesInPyramid(const Mat image, std::vector<Mat> &pyrs, std::vector<std::vector<Mat> > &patches_array, std::vector<std::vector<Point> > &coordinates_array,
                                      int octaves, int octaveLayers, double sigma,
                                      int number_patches , int number_of_target_layers);

    //sample patches in pyramids: version 2
    void randomSamplePatchesInPyramid(const Mat image, std::vector<Mat> &pyrs, std::vector<std::vector<Mat> > &patches_array, std::vector<std::vector<Point> > &coordinates_array,
                                       int octaves, int octaveLayers, double sigma,
                                       const std::vector<int> number_vector );

    //prepare number vector which indicate how much layers and how much samples for each layer you
    //want to sample
    void prepareNumberVectorForPyramidSampling(std::vector<int> &number_vector, int octaveLayers, int number_patches, int number_of_target_layers);

    //sample patches in a series of matirxes
    //notice: the length of the input vectors images should be larger than the length of number_vector
    void randomSamplePatchesInMultipleMatrixes(const std::vector<Mat> &images,
                                               std::vector<std::vector<Mat> > &patches_array,
                                               std::vector<std::vector<Point> > &coordinates_array,
                                          const std::vector<int> number_vector );

    //randomly sample given number of patches in a single image
    void randomSamplePatches(Mat const &original_image , std::vector<Mat> &patches,
                          std::vector<Point> &random_coordinates, int nbr_patches );

    //overload:the version that didn't return the random_coordinates
    void randomSamplePatches( Mat const &original_image , std::vector<Mat> &patches,
                          int nbr_patches );

    //randomly generate coordinates given the number of it
    void generateRandomCoordinates(Mat const &original_image,
                                std::vector<Point> &random_coordinates, int nbr_patches );

    //sample patches from one image according to the given coordinates
    void samplePatches(Mat const &original_image, const std::vector<Point> &random_coordinates,
                    std::vector<Mat> &patches );

    //detect overlapped patches, if cross correlation < threshold, mark its symbol as OVERLAP
    void detectOverlappedPatches(std::vector<Mat> &original_patches, std::vector<PATCH_TYPE> &symbol,
                                  double threshold , OVERLAP_METRIC metric);
    void detectOverlappedPatchesInPyramid(std::vector<std::vector<Mat> > &original_patches_array, std::vector<std::vector<PATCH_TYPE> > &symbols_array, double threshold, OVERLAP_METRIC metric);

    //detect flat patches, if score is < threshold, then this patch will be marked as flat patch
    void detectFlatPatches(std::vector<Mat> &original_patches, std::vector<PATCH_TYPE> &symbol, double threshold, FLAT_METRIC metric);
    void detectFlatPatchesInPyramid(std::vector<std::vector<Mat> > &original_patches_array, std::vector<std::vector<PATCH_TYPE> > &symbols_array, double threshold, FLAT_METRIC metric);

    //draw patches in image according to given symbols
    //red for overlappes, blue for flats, green for positives
    static void drawFrameForPatchesInImage(Mat &image, const std::vector<Point> coordinates,
                                    const std::vector<PATCH_TYPE> overlappedPatchSymbols,
                                    const std::vector<PATCH_TYPE> flatPatchSymbols,
                                    bool showPositives, bool showOverlappes, bool showFlats,
                                    const int patchesToShow, const int patchSize);

private:
    PatchesParameters *parameters;

};

#endif // PATCH_H
