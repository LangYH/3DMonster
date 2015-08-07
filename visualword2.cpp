#include "visualword2.h"
#include "visualword.h"
#include "databasemanager.h"
#include <QSqlQuery>
#include <QElapsedTimer>
#include <QVariant>
#include <assert.h>
#include <QDir>
#include "imtools.h"
#include <algorithm>
#include <QRegularExpression>
#include <queue>

#define FILE_NAME_DESCRIPTOR_NATURAL_SET1 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfN1.yaml"
#define FILE_NAME_DESCRIPTOR_NATURAL_SET2 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfN2.yaml"
#define CLUSTER_STORING_PATH "/home/lang/dataset/NYUDataset/visual_word_with_hard_negative_mining"

class PatchInfo{
public:
    int source_image_id;
    double scale_factor;
    double svm_score;
    Point coordinate;
    int size;
    Mat image_patch;
    Mat depth_patch;

    //friend bool operator!=( const PatchInfo &patch1, const PatchInfo &patch2 );
    bool operator < ( const PatchInfo &y ) const
    {
        return svm_score > y.svm_score;
    }
};

bool operator!=( const PatchInfo &patch1, const PatchInfo &patch2 )
{
    return patch1.source_image_id != patch2.source_image_id
            || patch1.coordinate != patch2.coordinate
            || patch1.scale_factor != patch2.scale_factor;
}

bool comparePatchSVMScore( const PatchInfo &x, const PatchInfo &y )
{
    return x.svm_score > y.svm_score;
}

bool comparePatchSourceImageId( const PatchInfo &x, const PatchInfo &y )
{
    return x.source_image_id < y.source_image_id;
}

VisualWord2::VisualWord2()
{
    loadDataFromDatabase();
    hog_descr = new HOGDescriptor( cvSize(80,80), cvSize( 8, 8 ),
                                   cvSize( 8, 8 ), cvSize( 8, 8 ), 9  );
}

VisualWord2::~VisualWord2()
{
    delete hog_descr;
}

void VisualWord2::loadDataFromDatabase()
{
    DatabaseManager::loadVisualWordTrainingData( I1, I2, D1,D2,N1,N2);
}

void VisualWord2::trainVisualWordInitialization()
{
    //get all visual word IDs from database
    QSqlQuery query;
    QString command = "SELECT class_id"
                      " FROM visual_word_3"
                      " ORDER BY class_id";
    CV_Assert( query.exec( command ) );

    while( query.next() ){
        visual_word_IDs.push_back( query.value("class_id").toInt() );
    }

    //get the corresponding patches cluster for each visual word
    QString data_command = "SELECT class_path, image_path"
                           " FROM visual_word_3"
                           " WHERE class_id= :class_id;";
    for( unsigned i = 0; i < visual_word_IDs.size(); i++ ){
        int current_class_label = visual_word_IDs[i];
        query.prepare( data_command );
        query.bindValue(":class_id", current_class_label );
        assert( query.exec() );

        assert( query.first() );

        QString class_path = query.value("class_path").toString();
        QString image_path = query.value("image_path").toString();
        QString cluster_path = class_path + QDir::separator() + image_path;
        QDir cluster_dir( cluster_path );

        QStringList current_cluster;
        QStringList filters;
        filters += "*.png";
        foreach (QString file, cluster_dir.entryList( filters )) {
            QString path = cluster_path + QDir::separator() + file;
            current_cluster.push_back( path );
        }

        if( visual_word_patches.count( current_class_label ) == 0 ){
            visual_word_patches[current_class_label] = current_cluster;
        }
        if( visual_word_paths.count( current_class_label ) == 0 ){
            visual_word_paths[current_class_label] = class_path;
        }
    }
    //load negative descriptor for training
    FileStorage fs;
    if( fs.open( FILE_NAME_DESCRIPTOR_NATURAL_SET1, FileStorage::READ ) ){
        fs["N1"] >> negative_descriptor_1;
    }
    fs.release();

    if( fs.open( FILE_NAME_DESCRIPTOR_NATURAL_SET2, FileStorage::READ ) ){
        fs["N2"] >> negative_descriptor_2;
    }
    fs.release();
}

void VisualWord2::kmeansInitialization()
{
    QSqlQuery query;
    QString command = "SELECT kmeans_class_label "
                      " FROM nyu_depth_patches "
                      " GROUP BY kmeans_class_label "
                      " HAVING COUNT(*) > 2 AND kmeans_class_label > 6369 "
                      " ORDER BY kmeans_class_label ; ";
    CV_Assert( query.exec( command ) );

    while( query.next() ){
        init_class_label.push_back( query.value("kmeans_class_label").toInt() );
    }

    QString data_command = "SELECT patch_path, patch_name "
                           " FROM nyu_depth_patches "
                           " WHERE kmeans_class_label = :label ;";
    for( unsigned i = 0; i < init_class_label.size(); i++ ){
        int current_class_label = init_class_label[i];
        query.prepare( data_command );
        query.bindValue(":label", current_class_label );
        assert( query.exec() );

        QStringList current_cluster;
        while( query.next() ){
            QString patch_path = query.value("patch_path").toString() + QDir::separator()
                            + query.value("patch_name").toString();
            current_cluster.push_back(patch_path);
        }
        if( clusters.count( current_class_label ) == 0 ){
            clusters[current_class_label] = current_cluster;
        }
        if( cluster_svm_scores.count( current_class_label == 0 ) ){
            cluster_svm_scores[current_class_label] = -10.0;
        }
    }


    //load negative descriptor for training
    FileStorage fs;
    if( fs.open( FILE_NAME_DESCRIPTOR_NATURAL_SET1, FileStorage::READ ) ){
        fs["N1"] >> negative_descriptor_1;
    }
    fs.release();

    if( fs.open( FILE_NAME_DESCRIPTOR_NATURAL_SET2, FileStorage::READ ) ){
        fs["N2"] >> negative_descriptor_2;
    }
    fs.release();

}

static void getSVMDetectorForHOG(MySVM *svm, std::vector<float> &myDetector);

void VisualWord2::detectTopPatches( MySVM &svm, const QStringList &target_depth_list,
                                    const QStringList &target_image_list,
                                   std::vector<PatchInfo> &depth_patches, int m)
{
    std::vector<PatchInfo>().swap( depth_patches );

    std::vector<float> myDetector;
    getSVMDetectorForHOG( &svm, myDetector );

    HOGDescriptor myHOG(Size(80,80),Size(8,8),Size(8,8), Size(8,8), 9 );
    myHOG.setSVMDetector( myDetector );

    std::priority_queue<PatchInfo> patch_queue;
    //fill heap with m false patch with high negative score;
    for( int i = 0; i < m; i++ ){
        PatchInfo tmp;
        tmp.svm_score = -10.0;
        patch_queue.push( tmp );
    }

    QRegularExpression re( "\\d+");
    for( int i = 0; i < target_depth_list.size(); i++ ){
        std::vector<Point> founded_position;
        std::vector<double> founded_score;

        Mat depth = imread( target_depth_list[i].toLocal8Bit().data(),
                            CV_LOAD_IMAGE_GRAYSCALE );

        Mat image = imread( target_image_list[i].toLocal8Bit().data() );

        myHOG.detect( depth, founded_position, founded_score,
                      -0.8, Size(8,8), Size(0,0) );

        int image_id = re.match(target_depth_list[i]).captured(0).toInt();
        for( unsigned k = 0; k < founded_position.size(); k++ ){
            if( founded_score[k] <= patch_queue.top().svm_score ){
                continue;
            }
            else{
                PatchInfo info;
                info.source_image_id = image_id;
                info.scale_factor = 1.0;
                info.svm_score = founded_score[k];
                info.coordinate = founded_position[k];
                info.image_patch = image(
                                        Rect( founded_position[k].x, founded_position[k].y, 80, 80 ) ).clone();
                info.depth_patch= depth(
                                        Rect( founded_position[k].x, founded_position[k].y, 80, 80 ) ).clone();
                patch_queue.pop();
                patch_queue.push( info );
            }
        }
    }

    while( !patch_queue.empty() ){
        depth_patches.push_back( patch_queue.top() );
        patch_queue.pop();
    }

    std::reverse( depth_patches.begin(), depth_patches.end() );

}

void VisualWord2::train()
{
    QElapsedTimer timer;
    timer.start();

    kmeansInitialization();
    assert( init_class_label.size() != 0 );


    for( unsigned i = 0; i < 8; i++ ){
        MySVM svm;
        trainOneSingleVisualWord( init_class_label[i], 5, svm );
    }

    std::cout << "Train 8 visual word, using time: " <<
                 QString::number( timer.elapsed() / 1000.0 ).toInt() << std::endl;

}

void VisualWord2::parallelTrain()
{

    kmeansInitialization();
    assert( init_class_label.size() != 0 );

#pragma omp parallel
    {
#pragma omp for schedule(dynamic,1) nowait
        for( unsigned i = 0; i < init_class_label.size(); i++ ){
            MySVM svm;
            trainOneSingleVisualWord( init_class_label[i], 5, svm );
        }

    }

    //updateDataBase();

}

void VisualWord2::parallelTrainVisualWordDetector()
{
    trainVisualWordInitialization();
    assert( visual_word_IDs.size() != 0 );
    QString save_svm_path = "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/svmData/visual_word_detectors/";

#pragma omp parallel
    {
#pragma omp for schedule( dynamic, 1 ) nowait
        for( unsigned i = 0; i < visual_word_IDs.size();i++){
            MySVM svm;
            trainVisualWordDetector( visual_word_IDs[i], 2, svm );
            QString svm_save_path = save_svm_path + "svm_detector_"
                            + QString::number(visual_word_IDs[i]) + ".txt";
            svm.save(svm_save_path.toLocal8Bit().data());
        }
    }

}

bool VisualWord2::trainOneSingleVisualWord( int class_label, int iterations, MySVM &svm )
{

    std::vector<PatchInfo> detect_patches_D1;
    //as a start point, use the kmeans result as the initial cluster
    getPositiveSamplesForGivenID( class_label,  detect_patches_D1 );
    std::vector<PatchInfo> detect_patches_D2;

    int iter = 0;
    std::vector<PatchInfo> found_patches;
    while( iter < iterations ){
        //stage one
        //training example: detect_patches_D1, N1
        //CV examples: D2, detect_patches_D2
        trainOneSVMDetector( detect_patches_D1, N1, STAGE_ONE, svm );

        detectTopPatches( svm, D2, I2, found_patches, 8 );

        if( found_patches.size() < 5 ){
            return false;
        }

        if( isEqualPatchesCluster( found_patches, detect_patches_D2 ) ){
            break;
        }
        else{
            detect_patches_D2.swap( found_patches );
        }

        //stage two
        //training examples: detect_patches_D2, N2,
        //CV examples: D1, detect_patches_D1
        trainOneSVMDetector( detect_patches_D2, N2, STAGE_TWO, svm );

        detectTopPatches( svm, D1, I1, found_patches, 8 );

        if( found_patches.size() < 5 ){
            return false;
        }

        if( isEqualPatchesCluster( found_patches, detect_patches_D1 ) ){
            break;
        }
        else{
            detect_patches_D1.swap( found_patches );
        }

        iter++;
    }

    storeClusters( class_label, detect_patches_D1, detect_patches_D2 );
    return true;
}

void VisualWord2::getPositiveSamplesForGivenID( int class_label,
                                                std::vector<PatchInfo> &detect_depth_patches_D1 )
{
    std::vector<PatchInfo>().swap( detect_depth_patches_D1 );

    for( int i = 0; i < clusters[class_label].size(); i++ ){
        PatchInfo current_patch;
        current_patch.source_image_id = -1;
        current_patch.depth_patch = imread( clusters[class_label][i].toLocal8Bit().data() );
        current_patch.image_patch = Mat();
        current_patch.scale_factor = 1.0;
        current_patch.coordinate = Point( -1, -1 );
        current_patch.svm_score = -10.0;

        detect_depth_patches_D1.push_back( current_patch );
    }
}

void VisualWord2::trainVisualWordDetector( int class_ID,
                                           int iteration,
                                           MySVM &svm )
{
    QStringList negative_image_list = N2;
    //get the positive sample patches
    std::vector<Mat> positive_patches;
    foreach( QString path, visual_word_patches[class_ID] ){
        Mat patch = imread( path.toLocal8Bit().data() );
        positive_patches.push_back(patch);
    }

    Mat positive_descriptor_Mat;
    imtools::computeHOGDescriptorsMat( positive_descriptor_Mat, positive_patches,
                                       hog_descr );

    Mat negative_descriptor_Mat;
    getNegativeDescriptorForGivenStage( STAGE_FULL, negative_descriptor_Mat );

    svmTrain( positive_descriptor_Mat, negative_descriptor_Mat, svm );

    //hard examples mining
    for( int i = 0; i < iteration; i++ ){
        std::vector<Mat> hard_examples;
        getHardExmaples( svm, negative_image_list, hard_examples );

        if( hard_examples.size() > 0 ){
            trainOneSVMDetectorWithHardExamples( positive_descriptor_Mat, negative_descriptor_Mat,
                                                 hard_examples, hog_descr, svm );
        }

        std::vector<Mat>().swap( hard_examples );
    }

}

void VisualWord2::trainOneSVMDetector( const std::vector<PatchInfo> &detect_depth_patches,
                          const QStringList &negative_image_list,
                          CROSS_VALIDATION_SYMBOL cv_symbol, MySVM &svm )
{
    Mat positive_descriptor_Mat;

    computeHOGDescriptorsMat( detect_depth_patches, positive_descriptor_Mat,
                              hog_descr);

    Mat negative_descriptor_Mat;
    getNegativeDescriptorForGivenStage( cv_symbol, negative_descriptor_Mat );

    svmTrain( positive_descriptor_Mat, negative_descriptor_Mat, svm );


    std::vector<Mat> hard_examples;
    getHardExmaples( svm, negative_image_list, hard_examples );

    if( hard_examples.size() > 0 ){
        trainOneSVMDetectorWithHardExamples( positive_descriptor_Mat, negative_descriptor_Mat,
                                             hard_examples, hog_descr, svm );
    }

    std::vector<Mat>().swap( hard_examples );
}

void VisualWord2::computeHOGDescriptorsMat( const std::vector<PatchInfo> &detect_depth_patches,
                               Mat &descriptorMat, const HOGDescriptor *hogDesr)
{
    //compute all descriptors of training images
    //store in descriptorMat, for each row of it is a descriptor of one image
    descriptorMat.create( detect_depth_patches.size(), 900, CV_32FC1 );
    int nr = descriptorMat.rows;
    int nc = descriptorMat.cols;
    for( int j=0; j<nr; j++ ){
        float *data = descriptorMat.ptr<float>(j);

        Mat image = detect_depth_patches.at(j).depth_patch.clone();

        if( image.channels() == 3 ){
            cvtColor( image, image, CV_BGR2GRAY );
        }

        std::vector<float> descr;
        hogDesr->compute( image, descr, Size(0, 0 ), Size( 0, 0 ) );
        for( int i = 0; i < nc; i++ ){
            *data++ = descr[i];
        }
    }

}

void VisualWord2::getNegativeDescriptorForGivenStage( CROSS_VALIDATION_SYMBOL cv_symbol,
                                         Mat &negative_descriptor_Mat )
{
    if( cv_symbol == STAGE_ONE ){
        negative_descriptor_Mat = negative_descriptor_1;
    }
    else if( cv_symbol == STAGE_TWO ){
        negative_descriptor_Mat = negative_descriptor_2;
    }
    else if( cv_symbol == STAGE_FULL ){
        negative_descriptor_Mat = negative_descriptor_1;
    }
    else{
        return;
    }
}

void VisualWord2::svmTrain( const Mat &positive_descriptor_Mat, const Mat &negative_descriptor_Mat,
               MySVM &svm )
{
    CvSVMParams svm_params;
    svm_params.svm_type = SVM::C_SVC;
    svm_params.C = 0.1;
    svm_params.kernel_type = SVM::LINEAR;
    svm_params.term_crit = TermCriteria( CV_TERMCRIT_ITER, 12, 1e-6 );


    Mat training_data_Mat;
    vconcat( positive_descriptor_Mat, negative_descriptor_Mat, training_data_Mat );

    Mat labels_Mat( positive_descriptor_Mat.rows + negative_descriptor_Mat.rows,
                    1, CV_32FC1, Scalar(-1.0) );
    labels_Mat( cv::Rect( 0, 0, 1, positive_descriptor_Mat.rows ) ).setTo( Scalar(1.0) );

    svm.train( training_data_Mat, labels_Mat, Mat(), Mat(), svm_params );
}

void VisualWord2::trainOneSVMDetectorWithHardExamples( const Mat &positive_descriptor_Mat,
                                          const Mat &negative_descriptor_Mat,
                                          const std::vector<Mat> &hard_examples,
                                          HOGDescriptor *hog_descr,
                                          MySVM &svm )
{
    CvSVMParams svm_params;
    svm_params.svm_type = SVM::C_SVC;
    svm_params.C = 0.1;
    svm_params.kernel_type = SVM::LINEAR;
    svm_params.term_crit = TermCriteria( CV_TERMCRIT_ITER, 12, 1e-6 );

    Mat hard_examples_descriptor_Mat;
    imtools::computeHOGDescriptorsMat( hard_examples_descriptor_Mat, hard_examples,
                                       hog_descr );

    Mat training_data_Mat;
    vconcat( positive_descriptor_Mat, negative_descriptor_Mat, training_data_Mat );
    vconcat( training_data_Mat, hard_examples_descriptor_Mat, training_data_Mat );

    Mat labels_Mat( training_data_Mat.rows, 1, CV_32FC1, Scalar(-1.0) );
    labels_Mat( cv::Rect( 0, 0, 1, positive_descriptor_Mat.rows ) ).setTo(Scalar(1.0));

    svm.train( training_data_Mat, labels_Mat, Mat(), Mat(), svm_params );

}


static void getSVMDetectorForHOG(MySVM *svm, std::vector<float> &myDetector)
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

void VisualWord2::getHardExmaples( MySVM &svm, const QStringList &negative_images_list,
                                   std::vector<Mat> &hard_examples )
{
    std::vector<Mat>().swap( hard_examples );

    std::vector<float> myDetector;
    getSVMDetectorForHOG( &svm, myDetector );

    HOGDescriptor myHOG(Size(80,80),Size(8,8),Size(8,8), Size(8,8), 9 );
    myHOG.setSVMDetector( myDetector );

    int nbr_negative_images_to_use = negative_images_list.size();
    for( int i = 0; i < nbr_negative_images_to_use; i += 3 ){
        std::vector<Rect> founded_rect;

        Mat image = imread( negative_images_list[i].toLocal8Bit().data(),
                            CV_LOAD_IMAGE_GRAYSCALE );

        myHOG.detectMultiScale( image, founded_rect, -1.0, Size(8,8), Size(0,0),
                                1.5, 2 );

        for( unsigned i = 0; i < founded_rect.size(); i++ ){
            Rect r = founded_rect[i];

            if( r.x < 0 || r.y < 0 || r.x + r.width > image.cols
                    || r.y + r.height > image.rows )
                continue;

            Mat hard_example = image( r ).clone();
            cv::resize( hard_example, hard_example, Size(80,80) );
            hard_examples.push_back( hard_example );
        }
    }
}

bool VisualWord2::isEqualPatchesCluster( std::vector<PatchInfo> &found_depth_patches,
                                         std::vector<PatchInfo> &detect_depth_patches )
{
    if( found_depth_patches.size() != detect_depth_patches.size() ){
        return false;
    }

    std::sort( found_depth_patches.begin(), found_depth_patches.end(),
               comparePatchSourceImageId );
    std::sort( detect_depth_patches.begin(), detect_depth_patches.end(),
               comparePatchSourceImageId );

    for( unsigned i = 0; i < found_depth_patches.size(); i++ ){
        if( found_depth_patches[i] != detect_depth_patches[i] ){
            return false;
        }
    }

    return true;
}

void VisualWord2::storeClusters(  int target_class_label,
                                  const std::vector<PatchInfo> &detect_patches_D1,
                                  const std::vector<PatchInfo> &detect_patches_D2 )
{
    //prepare path to save image patch and depth patch
    QDir path_of_all_cluster( CLUSTER_STORING_PATH );
    assert( path_of_all_cluster.mkdir( "cluster_" + QString::number(target_class_label)) );

    QString path_of_current_cluster = path_of_all_cluster.path() + QDir::separator() +
            "cluster_" + QString::number( target_class_label );

    assert( QDir( path_of_current_cluster ).mkdir("image_patches")) ;
    assert( QDir( path_of_current_cluster ).mkdir("depth_patches")) ;

    QString path_of_saving_images = path_of_current_cluster + QDir::separator() + "image_patches";
    QString path_of_saving_depths = path_of_current_cluster + QDir::separator() + "depth_patches";

    //And weight average of depth patch
    double svm_weight_sum = 0.0;
    Mat sum_depth_Mat( detect_patches_D1[0].depth_patch.rows, detect_patches_D1[1].depth_patch.cols,
            CV_32FC1, Scalar(0.0) );

    int counts = 0;
    for( unsigned i = 0; i < detect_patches_D1.size(); i++ ){
        svm_weight_sum += detect_patches_D1[i].svm_score;
        Mat tmp;
        detect_patches_D1[i].depth_patch.convertTo( tmp, CV_32FC1 );
        sum_depth_Mat += tmp * detect_patches_D1[i].svm_score;

        QString save_image_patch_path = path_of_saving_images + QDir::separator() +
                QString::number( counts ) + ".png";
        QString save_depth_patch_path = path_of_saving_depths + QDir::separator() +
                QString::number( counts ) + ".png";
        imwrite( save_image_patch_path.toLocal8Bit().data(), detect_patches_D1[i].image_patch );
        imwrite( save_depth_patch_path.toLocal8Bit().data(), detect_patches_D1[i].depth_patch );
        counts++;
    }

    for( unsigned i = 0; i < detect_patches_D2.size(); i++ ){
        svm_weight_sum += detect_patches_D2[i].svm_score;
        Mat tmp;
        detect_patches_D1[i].depth_patch.convertTo( tmp, CV_32FC1 );
        sum_depth_Mat += tmp * detect_patches_D2[i].svm_score;

        QString save_image_patch_path = path_of_saving_images + QDir::separator() +
                QString::number( counts ) + ".png";
        QString save_depth_patch_path = path_of_saving_depths + QDir::separator() +
                QString::number( counts ) + ".png";
        imwrite( save_image_patch_path.toLocal8Bit().data(), detect_patches_D2[i].image_patch );
        imwrite( save_depth_patch_path.toLocal8Bit().data(), detect_patches_D2[i].depth_patch );
        counts++;
    }

    sum_depth_Mat /= svm_weight_sum;
    cluster_svm_scores[target_class_label] = svm_weight_sum;
    QString save_depth_path = path_of_current_cluster + QDir::separator() + "average_depth.png";
    cv::normalize( sum_depth_Mat, sum_depth_Mat, 0, 255, NORM_MINMAX, CV_8UC1 );
    imwrite( save_depth_path.toLocal8Bit().data(), sum_depth_Mat);
}

void VisualWord2::updateDataBase()
{
    QSqlQuery query;
    QString insert_command = "INSERT INTO visual_word_3 ( class_id, class_path, image_path, depth_path, svm_score, depth_average_patch ) "
                             " VALUES (:class_id, :class_path, :image_path, :depth_path, :svm_score, :average_patch ) ; ";

    std::map<int,double>::const_iterator iter;
    assert(QSqlDatabase::database().transaction());
    for( iter = cluster_svm_scores.begin(); iter != cluster_svm_scores.end();
         iter++ ){
        int target_class_label = iter->first;
        QString path_of_current_cluster = QString(CLUSTER_STORING_PATH) + QDir::separator() +
                        "cluster_" + QString::number( target_class_label );
        query.prepare( insert_command );
        query.bindValue( ":class_id", target_class_label );
        query.bindValue( ":class_path", path_of_current_cluster );
        query.bindValue( ":image_path", "image_patches");
        query.bindValue( ":depth_path", "depth_patches");
        query.bindValue( ":svm_score", iter->second );
        query.bindValue( ":average_patch", "average_depth.png" );
        assert( query.exec() );
    }
    assert(QSqlDatabase::database().commit());
}
