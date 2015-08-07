#include "visualworddictionary.h"
#include <QDir>
#include <QSqlQuery>
#include <QElapsedTimer>
#include "imtools.h"
#include <QVariant>

#define FILE_NAME_DESCRIPTOR_NYU_SET1 "HOGData/descriptorMatOfD1.yaml"
#define FILE_NAME_DESCRIPTOR_NATURAL_SET1 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfN1.yaml"
#define FILE_NAME_DESCRIPTOR_NATURAL_SET2 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfN2.yaml"
#define SVM_CLASSIFIER_PATH "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/svmData/visual_word_classifiers"
#define SVM_CLASSIFIER_PATH_2 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/svmData/visual_word_classifiers_2"


VisualWordDictionary::VisualWordDictionary()
{
    hog_descr = new HOGDescriptor( cvSize(80,80), cvSize( 8, 8 ),
                                   cvSize( 8, 8 ), cvSize( 8, 8 ), 9  );
}

VisualWordDictionary::~VisualWordDictionary()
{
    delete hog_descr;
    cleanAllSVMClassifiers();
}

void VisualWordDictionary::loadNaturalImages()
{
    //get all natural images
    QSqlQuery query;
    QString command = "SELECT image_path, image_name "
            " FROM natural_image "
            " WHERE id % 6 = 0 "
            " ORDER BY id "
            " LIMIT 1000; ";

    assert( query.exec(command) );

    while( query.next() ){
        QString path = query.value(0).toString();
        QString name = query.value(1).toString();
        QString full_path = path + QDir::separator() + name;
        natural_images_list.push_back( full_path );
    }
}

bool VisualWordDictionary::trainDictionary()
{
    return trainMultipleSVMClassifier();
}

void VisualWordDictionary::prepareNegativeSamplesForMultipleSVMTraining(Mat &negativeDescriptorMat)
{


    QString command = "SELECT class_path, image_path "
            " FROM visual_word"
            " WHERE available = 1 AND svm_score <= 7.5 "
            " ORDER BY class_id ;";
    QSqlQuery query;
    assert( query.exec(command) );


    while( query.next() ){
        //get all patches of one class
        QString path = query.value(0).toString() + QDir::separator() +
                        query.value(1).toString();
        QDir target_dir( path );
        QStringList filters;
        filters += "*.png";
        QStringList positiveSamplesList;
        foreach (QString file, target_dir.entryList(filters,QDir::Files)) {
            QString full_path = path + QDir::separator() + file;
            positiveSamplesList.push_back( full_path );
        }
        //*****************************

        //add the descriptors and labels to training Mat
        Mat positive_descriptor_mat;
        imtools::computeHOGDescriptorsMat( positive_descriptor_mat,
                                           positiveSamplesList, hog_descr );

        if( negativeDescriptorMat.empty() )
            negativeDescriptorMat = positive_descriptor_mat.clone();
        else
            vconcat( negativeDescriptorMat, positive_descriptor_mat, negativeDescriptorMat );

    }

}

bool VisualWordDictionary::trainMultipleSVMClassifier()
{
    QElapsedTimer timer;
    timer.start();
    loadNaturalImages();
    //prepare negative samples, which are from natural image set
    Mat negative_descriptor_Mat;
    prepareNegativeSamplesForMultipleSVMTraining( negative_descriptor_Mat );
    //FileStorage fs;
    //if( fs.open( FILE_NAME_DESCRIPTOR_NATURAL_SET1, FileStorage::READ ) ){
    //    fs["N1"] >>  negative_big_mat ;
    //}

    //SVM parameter setting
    CvMat *weights = cvCreateMat( 2, 1, CV_32FC1 );
    cvmSet( weights, 0, 0, 1.0 );
    cvmSet( weights, 1, 0, 0.1 );
    CvSVMParams svm_params;
    svm_params.svm_type = SVM::C_SVC;
    svm_params.C = 1.0;
    svm_params.class_weights = weights;
    svm_params.kernel_type = SVM::LINEAR;
    svm_params.term_crit = TermCriteria( CV_TERMCRIT_ITER, 30, 1e-6 );


    //train a SVM classifier for every visual word whose svm_score in depth space
    //is bigger than 7.5
    std::map<int,Mat> positive_samples;
    getAllPositiveSample( positive_samples, 7.5 );

    std::map<int,Mat>::const_iterator iter;
    for( iter = positive_samples.begin(); iter != positive_samples.end(); iter++ ){
        //for those which are not the current class, add them the negative samples
        Mat other_class_mat;
        std::map<int, Mat>::const_iterator iter_other;
        for( iter_other = positive_samples.begin(); iter_other != positive_samples.end();
             iter_other++ )
            if( iter_other->first != iter->first ){
                if( other_class_mat.empty() )
                    other_class_mat = iter_other->second.clone();
                else
                    vconcat( other_class_mat, iter_other->second,
                             other_class_mat );
            }
        //****************************************************************************

        Mat negative_big_mat;
        vconcat( negative_descriptor_Mat, other_class_mat, negative_big_mat );

        Mat positive_labels_mat( iter->second.rows, 1, CV_32FC1, Scalar(1.0) );
        Mat negative_labels_mat( negative_big_mat.rows, 1, CV_32FC1, Scalar(-1.0));

        //now we can have the training examples and labels for SVM trainging
        Mat samples_matrixes, labels_matrixes;
        vconcat( iter->second, negative_big_mat, samples_matrixes );
        vconcat( positive_labels_mat, negative_labels_mat, labels_matrixes );

        //svm training...
        QString classifier_save_path = "svmData/visual_word_classifiers_2/visual_word_"
                        + QString::number( iter->first ) + ".txt";
        MySVM svm;
        svm.train( samples_matrixes, labels_matrixes, Mat(), Mat(), svm_params );

        //reinforcement learning
        std::vector<Mat> hard_examples;

        QElapsedTimer timer;
        timer.start();
        getHardExmaples( svm, hard_examples );
        std::cout << "Using time: " << timer.elapsed() / 1000.0
                  << " seconds " << std::endl;
        std::cout << hard_examples.size() << " hard examples have been found " << std::endl;

        if( hard_examples.size() > 0 ){
            Mat hard_examples_descriptor_Mat;
            imtools::computeHOGDescriptorsMat( hard_examples_descriptor_Mat,
                                               hard_examples, hog_descr );

            Mat hard_examples_labels_Mat( hard_examples_descriptor_Mat.rows, 1,
                                          CV_32FC1, Scalar(-1.0 ) );
            Mat samples_matrixes2, labels_matrixes2;
            vconcat( samples_matrixes, hard_examples_descriptor_Mat, samples_matrixes2 );
            vconcat( labels_matrixes, hard_examples_labels_Mat, labels_matrixes2 );
            svm.train( samples_matrixes2, labels_matrixes2, Mat(), Mat(), svm_params );

            svm.save( classifier_save_path.toLocal8Bit().data() );

            //svm.train_auto( samples_matrixes, labels_matrixes, Mat(), Mat(), svm_params, 2,
            //                CvSVM::get_default_grid(CvSVM::C),
            //                CvSVM::get_default_grid(CvSVM::GAMMA),
            //                CvSVM::get_default_grid(CvSVM::P),
            //                CvSVM::get_default_grid(CvSVM::NU),
            //                CvSVM::get_default_grid(CvSVM::COEF),
            //                CvSVM::get_default_grid(CvSVM::DEGREE),
            //                true);
            //CvSVMParams params_re = svm.get_params();
            //float C = params_re.C;
            //float P = params_re.p;
            //float gamma = params_re.gamma;
            //std::cout << "C:" << C << "		P:" << P << " 	gamma:" << gamma << std::endl;
        }
    }
    std::cout << "Using time: " << timer.elapsed() / 1000.0
              << " seconds " << std::endl;

    return true;

}

void VisualWordDictionary::getAllPositiveSample( std::map<int,Mat> &positive_samples,
                                                 double score_threshold )
{
    //train a SVM classifier for every visual word whose svm_score in depth space
    //is bigger than 7.5
    QString command = "SELECT class_id, class_path, image_path "
            " FROM visual_word_2"
            " WHERE available = 1 AND svm_score > :threshold "
            " ORDER BY class_id ;";
    QSqlQuery query;
    query.prepare( command );
    query.bindValue(":threshold", score_threshold );
    assert( query.exec() );

    //get all samples for each positive class
    while( query.next() ){
        //get all patches of one class
        int class_id = query.value(0).toInt();
        QString path = query.value(1).toString() + QDir::separator() +
                        query.value(2).toString();
        QDir target_dir( path );
        QStringList filters;
        filters += "*.png";
        QStringList positiveSamplesList;
        foreach (QString file, target_dir.entryList(filters,QDir::Files)) {
            QString full_path = path + QDir::separator() + file;
            positiveSamplesList.push_back( full_path );
        }
        //*****************************

        //add the descriptors and labels to training Mat
        Mat positive_descriptor_mat;
        imtools::computeHOGDescriptorsMat( positive_descriptor_mat,
                                           positiveSamplesList, hog_descr );

        if( positive_samples.count(class_id) == 0 )
            positive_samples[class_id] = positive_descriptor_mat;
    }
}

void VisualWordDictionary::getHardExmaples( MySVM &svm, std::vector<Mat> &hard_examples )
{
    std::vector<Mat>().swap( hard_examples );
    //detect each natural image ( containing no positive )
    //return all detections ( i.e. false positive )

    std::vector<float> myDetector;
    getSVMDetectorForHOG( &svm, myDetector );

    HOGDescriptor myHOG(Size(80,80),Size(8,8),Size(8,8), Size(8,8), 9 );
    myHOG.setSVMDetector( myDetector );
    for( int i = 0; i < natural_images_list.size(); i++ ){
        std::vector<Rect> founded_rect;

        Mat image = imread( natural_images_list[i].toLocal8Bit().data(),
                            CV_LOAD_IMAGE_GRAYSCALE );

        //searchImageForGivenWord( image, svm,
        //                         founded_rect, scores );
        myHOG.detectMultiScale( image, founded_rect, -0.8, Size(8,8), Size(0,0),
                                1.3, 2 );

        for( unsigned i = 0; i < founded_rect.size(); i++ ){
            Rect r = founded_rect[i];
            if( r.x < 0 )
                r.x = 0;
            if( r.y < 0 )
                r.y = 0;
            if( r.x + r.width > image.cols )
                r.width = image.cols - r.x;
            if( r.y + r.height > image.rows )
                r.height = image.rows - r.y;

            Mat hard_example = image( r ).clone();
            cv::resize( hard_example, hard_example, Size(80,80) );
            hard_examples.push_back( hard_example );
        }
    }
}

void VisualWordDictionary::getCanonicalPatchesForGiveRects( const Mat &image,
                                                            const std::vector<Rect> founded_rect,
                                                            std::vector<Mat> &patches )
{
    for( unsigned i = 0; i < founded_rect.size(); i++ ){
        Mat roi( image, founded_rect[i] );
        Mat canonical_patch;
        cv::resize( roi, canonical_patch, Size( 80, 80 ) );
        patches.push_back( canonical_patch );
    }
}

void VisualWordDictionary::cleanAllSVMClassifiers()
{
    std::map<int, MySVM*>::const_iterator iter;
    for( iter = classifiers.begin(); iter != classifiers.end(); iter++ ){
        delete iter->second;
    }
}

bool VisualWordDictionary::loadAllSVMClassifiers()
{
    classifiers.clear();

    QDir svm_dir( SVM_CLASSIFIER_PATH_2 );
    QStringList filters;
    filters += "*.txt";
    foreach (QString classifier_name, svm_dir.entryList(filters, QDir::Files )) {
        //get class id
        QString file_name = QFileInfo( classifier_name ).baseName();
        int class_id = file_name.split("_" )[2].toInt();
        QString full_path = svm_dir.path() + QDir::separator() + classifier_name;

        //load classifer
        if( classifiers.count(class_id) == 0 ){
            classifiers[class_id] = new MySVM();
            classifiers[class_id]->load( full_path.toLocal8Bit().data() );
        }
    }

    if( classifiers.empty() ){
        return false;
    }
    else{
        return true;
    }
}

void VisualWordDictionary::getSVMDetectorForHOG(MySVM *svm, std::vector<float> &myDetector)
{
    std::vector<float>().swap( myDetector );
    int descriptor_dims = svm->get_var_count();
    int support_vector_num = svm->get_support_vector_count();

    Mat alpha_Mat = Mat::zeros( 1, support_vector_num, CV_32FC1 );
    Mat support_vector_Mat = Mat::zeros( support_vector_num, descriptor_dims, CV_32FC1 );
    Mat result_Mat = Mat::zeros( 1, descriptor_dims, CV_32FC1 );

    //get support vector data
    for( int i = 0; i < support_vector_num; i++ ){
        const float* pSVData = svm->get_support_vector(i);
        for( int j = 0; j < descriptor_dims; j++ ){
            support_vector_Mat.at<float>(i,j) = pSVData[j];
        }
    }

    //get alpha data
    double* p_alpha_data = svm->get_alpha_vector();
    for( int i = 0; i < support_vector_num; i++ ){
        alpha_Mat.at<float>(0,i) = p_alpha_data[i];
    }

    result_Mat = -1 * alpha_Mat * support_vector_Mat;

    for( int i = 0; i < descriptor_dims; i++ ){
        myDetector.push_back( result_Mat.at<float>(0,i) );
    }

    //get the intercept value
    myDetector.push_back( svm->get_rho() );
}

Mat VisualWordDictionary::searchForDepthWithGiveID( int class_id )
{
    QString command = "SELECT class_path, depth_average_patch "
                      " FROM visual_word_2 "
                      " WHERE class_id = :class_id ;";
    QSqlQuery query;
    query.prepare(command);
    query.bindValue(":class_id", class_id );
    CV_Assert( query.exec() );

    if( query.size() == 0 )
        return Mat();
    query.first();
    QString depthmap_path = query.value(0).toString() + QDir::separator() +
                    query.value(1).toString();
    Mat depthmap = imread( depthmap_path.toLocal8Bit().data() , CV_LOAD_IMAGE_GRAYSCALE );

    return depthmap.clone();

}

int VisualWordDictionary::searchForId(const Mat &patch, double &score )
{
    int best_match_class_id = -1;
    double best_match_score = -10.0;
    std::map<int, MySVM*>::const_iterator iter;
    Mat temp;
    if( patch.channels() == 3 ){
        cvtColor( patch, temp, CV_BGR2GRAY );
    }
    else{
        temp = patch.clone();
    }
    cv::equalizeHist( temp, temp );
    //scan it with all SVM classifier
    for( iter = classifiers.begin(); iter != classifiers.end(); iter++ ){
        std::vector<float> descr;
        hog_descr->compute( temp, descr, Size(0, 0 ), Size( 0, 0 ) );
        float score = -( iter->second->predict( Mat(descr).t(), true ) );
        if( score > best_match_score ){
            best_match_score = score;
            best_match_class_id = iter->first;
        }
    }

    score = best_match_score;

    return best_match_class_id;
}

void VisualWordDictionary::searchImageForGivenWord(const Mat &image, MySVM &svm,
                                                       std::vector<Rect> &filtered_result,
                                                       std::vector<double> &filtered_scores )
{
    std::vector<float> myDetector;
    getSVMDetectorForHOG( &svm, myDetector );

    HOGDescriptor myHOG(Size(80,80),Size(8,8),Size(8,8), Size(8,8), 9 );
    myHOG.setSVMDetector( myDetector );

    std::vector<Rect> found;
    std::vector<double> scores;

    myHOG.detectMultiScale( image, found, scores, -0.8, Size(8,8), Size(0,0), 1.05, 2, false );

    for( unsigned i = 0; i < found.size(); i++ ){
        Rect r = found[i];
        if( r.x < 0 || r.y < 0 || ( r.x + r.width ) > image.cols
                || ( r.y + r.height ) > image.rows ){
            continue;
        }
        unsigned int j = 0;
        for( ; j < found.size(); j++ )
            if( j != i && ( r & found[j] ) == r )
                break;

        if( j == found.size() ){
            filtered_result.push_back( r );
            filtered_scores.push_back( scores[i]);
        }
    }
}

bool VisualWordDictionary::loadDictionary()
{
    loadAllSVMClassifiers();
    return true;
}
