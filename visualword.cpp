#include "visualword.h"
#include <QSqlQuery>
#include <QVariant>
#include <QDir>
#include <sys/time.h>
#include "imtools.h"
#include <QFile>

#define NUMBER_OF_DISCOVERY_SET1 46848
#define NUMBER_OF_DISCOVERY_SET2 46849
#define CROSS_VALIDATION_THRESHOLD 3
#define SVM_FIRING_THRESHOLD -1.0
#define NUMBER_OF_POSITIVE_SAMPLE_TO_KEEP 8
#define FILE_NAME_DESCRIPTOR_NYU_SET1 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfD1.yaml"
#define FILE_NAME_DESCRIPTOR_NYU_SET2 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfD2.yaml"
#define FILE_NAME_DESCRIPTOR_NATURAL_SET1 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfN1.yaml"
#define FILE_NAME_DESCRIPTOR_NATURAL_SET2 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfN2.yaml"
#define PATH_SVM_CLASSIFIERS "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/svmData"

VisualWord::VisualWord()
{
    m = 5;
    centroids = 0;
    kmeans_iterations = 0;
    training_iterations = 0;
    data_loaded = false;

    svm_params.svm_type = SVM::C_SVC;
    svm_params.C = 0.1;
    svm_params.kernel_type = SVM::LINEAR;
    svm_params.term_crit = TermCriteria( CV_TERMCRIT_ITER, 12, 1e-6 );

    hog_descr = new HOGDescriptor( cvSize(80,80), cvSize( 8, 8 ),
                                   cvSize( 8, 8 ), cvSize( 8, 8 ), 9  );
}

VisualWord::~VisualWord()
{
    delete hog_descr;
}

bool VisualWord::isDataLoaded()
{
    return data_loaded;
}

void VisualWord::setCentroids(int k)
{
    centroids = k;
}

void VisualWord::setKmeansIterations( int i )
{
    kmeans_iterations = i;
}

void VisualWord::setTrainingIterations(int i)
{
    training_iterations = i;
}


void VisualWord::train()
{
    //---------Step1: Initailization( mainly use kmeans )-----------------------------
    loadDataFromDatabase();
    kmeansInitialize();
    //----------Step2:Iteration process ( mainly use SVM and cross validation )--------------------

}

void VisualWord::loadDataFromDatabase()
{
    getDataForTrainingFromDatabase( D1, D2, N1, N2 );

    //if the N1 and N2 descriptor Mat haven't been compute , then compute them and save
    if( QFile::exists( FILE_NAME_DESCRIPTOR_NATURAL_SET1 )
                    && QFile::exists( FILE_NAME_DESCRIPTOR_NATURAL_SET2 ) ){
        ;
    }
    else {
        std::cout << "HOG Mat hasn't been computed, now computing..." << std::endl;
        FileStorage fs1, fs2;
        Mat descriptor_matrix_N1, descriptor_matrix_N2;
        computeDesriptorMat( N1, descriptor_matrix_N1 );
        computeDesriptorMat( N2, descriptor_matrix_N2 );

        if( fs1.open( FILE_NAME_DESCRIPTOR_NATURAL_SET1, FileStorage::WRITE ) ){
            fs1 << "N1" << descriptor_matrix_N1;
        }
        fs1.release();

        if( fs2.open( FILE_NAME_DESCRIPTOR_NATURAL_SET2, FileStorage::WRITE ) ){
            fs2 << "N2" << descriptor_matrix_N2;
        }
        fs2.release();

    }

    data_loaded = true;

    std::cout << "data loaded!" << std::endl;

}

int VisualWord::getDiscoverySamplesSize()
{
    return D1.length();
}

void VisualWord::kmeansInitialize()
{
    Mat descriptor_matrix_D1;
    computeDesriptorMat( D1, descriptor_matrix_D1 );

    if( centroids == 0 ){
        //centroids = D1.size() / 4
        setCentroids( D1.size() / 4 );
    }

    if( kmeans_iterations == 0 ){
        setKmeansIterations(3);
    }

    kmeans( descriptor_matrix_D1, centroids, bestlabels,
            TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0 ),
            kmeans_iterations, KMEANS_PP_CENTERS, centers );

    FileStorage fs;
    if( fs.open( "kmeans_result.yaml", FileStorage::WRITE ) ){
        fs << "bestlabels" << bestlabels;
        fs << "centers" << centers;
    }
    fs.release();

}

bool VisualWord::trainOneVisualWord( CvSVM &svm, const int init_class_label, const int iteration )
{
    int iter = 0;

    initDatabaseClassLabel();
    while( iter < iteration ){
        std::vector<int> D1_last_label = D1_label;
        std::vector<int> D2_last_label = D2_label;
        int fires = 0;
        //training step
        cleanClassLabel( STAGE_ONE );
        svmTrain( svm, init_class_label, STAGE_ONE );
        fires = svmDetect( svm, init_class_label, STAGE_ONE );
        if( fires < CROSS_VALIDATION_THRESHOLD )
            return false;

        if( D2_last_label == D2_label ){
            updateDatabase();
            return true;
        }

        keepTopResults( NUMBER_OF_POSITIVE_SAMPLE_TO_KEEP, init_class_label, STAGE_ONE );

        //cross validation step
        cleanClassLabel( STAGE_TWO );
        svmTrain( svm, init_class_label, STAGE_TWO );
        fires = svmDetect( svm, init_class_label, STAGE_TWO );
        if( fires < CROSS_VALIDATION_THRESHOLD )
            return false;

        if( D1_last_label == D1_label ){
            updateDatabase();
            return true;
        }

        keepTopResults( NUMBER_OF_POSITIVE_SAMPLE_TO_KEEP, init_class_label, STAGE_TWO );
        updateDatabase();

        iter++;

    }

    return true;

}

void VisualWord::svmTrain(CvSVM &svm, const int class_label, CROSS_VALIDATION_SYMBOL cv_symbol)
{

    //get training samples
    QStringList positive_list;
    Mat negativeDescriptorMat;
    FileStorage fs;
    if( cv_symbol == STAGE_ONE ){
        getPositiveSampleList( positive_list, class_label, STAGE_ONE );
        CV_Assert( positive_list.size() > 0 );
        if( fs.open( FILE_NAME_DESCRIPTOR_NATURAL_SET1, FileStorage::READ ) ){
            fs["N1"] >> negativeDescriptorMat;
        }

    } else {
        getPositiveSampleList( positive_list, class_label, STAGE_TWO );
        CV_Assert( positive_list.size() > 0 );
        if( fs.open( FILE_NAME_DESCRIPTOR_NATURAL_SET2, FileStorage::READ ) ){
            fs["N2"] >> negativeDescriptorMat;
        }
    }


    Mat positive_descriptor_matrix;
    computeDesriptorMat( positive_list, positive_descriptor_matrix );

    Mat training_data_Mat;
    vconcat( positive_descriptor_matrix, negativeDescriptorMat, training_data_Mat );

    Mat labels_Mat( positive_descriptor_matrix.rows + negativeDescriptorMat.rows,
                    1, CV_32FC1, Scalar(-1.0) );

    labels_Mat( cv::Rect( 0, 0, 1, positive_descriptor_matrix.rows ) ).setTo( Scalar(1.0) );

    svm.train( training_data_Mat, labels_Mat, Mat(), Mat(), svm_params );

}

int VisualWord::svmDetect(CvSVM &svm, const int class_label, CROSS_VALIDATION_SYMBOL cv_symbol)
{
    int fires = 0;

    if( cv_symbol == STAGE_ONE ){
        for( int i = 0; i < D2.size(); i++ ){
            //svm predict process
            Mat im = imread( D2[i].toLocal8Bit().data(),
                             CV_LOAD_IMAGE_GRAYSCALE );

            std::vector<float> descr;
            hog_descr->compute( im, descr, Size(0, 0 ), Size( 0, 0 ) );
            float score = -svm.predict( Mat(descr).t(), true );
            float response = svm.predict( Mat(descr).t() );

            //update label and score
            if( score > -1.0 ){
                //if score > -1.0, that's a fire
                D2_label.at(i) = class_label;
                D2_score.at(i) = score;
                fires++;
            }
        }
    }
    else{
        for( int i = 0; i < D1.size(); i++ ){
            //svm predict process
            Mat im = imread( D1[i].toLocal8Bit().data(),
                             CV_LOAD_IMAGE_GRAYSCALE );

            std::vector<float> descr;
            hog_descr->compute( im, descr, Size(0, 0 ), Size( 0, 0 ) );
            float score = -svm.predict( Mat(descr).t(), true );
            float response = svm.predict( Mat(descr).t() );

            //update label and score
            if( score > -1.0 ){
                D1_label.at(i) = class_label;
                D1_score.at(i) = score;
                fires++;
            }
        }
    }

    return fires;

}

void VisualWord::initDatabaseClassLabel()
{
    D1_label.assign( NUMBER_OF_DISCOVERY_SET1, -1 );
    D2_label.assign( NUMBER_OF_DISCOVERY_SET2, -1 );

    D1_score.assign( NUMBER_OF_DISCOVERY_SET1, -10.0 );
    D2_score.assign( NUMBER_OF_DISCOVERY_SET2, -10.0 );

    QSqlQuery query;
    QString reset_command = "SELECT kmeans_class_label "
                            " FROM nyu_depth_patches"
                            " WHERE id <= :dsize "
                            " ORDER BY id ; ";
    query.prepare( reset_command );
    query.bindValue( ":dsize", NUMBER_OF_DISCOVERY_SET1 );
    CV_Assert( query.exec() );

    int i = 0;
    while( query.next() ){
        D1_label.at(i) = query.value(0).toInt();
        i++;
    }

}
void VisualWord::updateDatabase()
{
    QString clean_database = "UPDATE nyu_depth_patches "
                             " SET test_label = -1, test_score = -10.0 ;";

    QString update_command = "UPDATE nyu_depth_patches "
                             " SET test_label = :label, test_score = :score "
                             " WHERE id = :id ;";

    QSqlQuery query;

    CV_Assert( query.exec( clean_database ) );

    CV_Assert( QSqlDatabase::database().transaction() );
    for( unsigned i = 0; i < D1_label.size(); i++ ){
        if( D1_label.at(i) != -1 ){
            query.prepare( update_command );
            query.bindValue( ":label", D1_label.at(i) );
            query.bindValue( ":score", D1_score.at(i) );
            query.bindValue( ":id", i + 1 );
            query.exec();
        }
    }

    for( unsigned i = 0; i < D2_label.size(); i++ ){
        if( D2_label.at(i) != -1 ){
            query.prepare( update_command );
            query.bindValue( ":label", D2_label.at(i) );
            query.bindValue( ":score", D2_score.at(i) );
            query.bindValue( ":id", NUMBER_OF_DISCOVERY_SET1 + i + 1 );
            query.exec();
        }
    }
    CV_Assert( QSqlDatabase::database().commit() );
}

void VisualWord::cleanClassLabel(CROSS_VALIDATION_SYMBOL cv_symbol)
{
    if( cv_symbol == STAGE_ONE ){
        D2_label.assign( D2_label.size(), -1 );
        D2_score.assign( D2_score.size(), -10.0 );
    }
    else{
        D1_label.assign( D1_label.size(), -1 );
        D1_score.assign( D1_score.size(), -10.0 );
    }
}

void VisualWord::getPositiveSampleList(QStringList &positive_list, const int class_label,
                                       CROSS_VALIDATION_SYMBOL cv_symbol)
{
    if( cv_symbol == STAGE_ONE ){
        for( unsigned i = 0; i < D1_label.size(); i++ ){
            if( D1_label.at(i) == class_label )
                positive_list.push_back( D1.at(i) );
        }
    }
    else{
        for( unsigned i = 0; i < D2_label.size(); i++ ){
            if( D2_label.at(i) == class_label )
                positive_list.push_back( D2.at(i) );
        }
    }
}

void VisualWord::keepTopResults(const int m, const int class_label, CROSS_VALIDATION_SYMBOL cv_symbol)
{
    //only keep the m most similar sample to target class
    if( cv_symbol == STAGE_ONE ){
        Mat score( D2_score );
        Mat sortedResult;
        sortIdx( score, sortedResult, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING );
        D2_label.assign( NUMBER_OF_DISCOVERY_SET2, -1 );
        for( int i = 0; i < m; i++ ){
            int index = sortedResult.at<int>( i, 0 );
            if( D2_score.at( index ) > -10.0){
                D2_label.at( index ) = class_label;
            }
        }
    }
    else{
        Mat score( D1_score );
        Mat sortedResult;
        sortIdx( score, sortedResult, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING );
        D1_label.assign( NUMBER_OF_DISCOVERY_SET2, -1 );
        for( int i = 0; i < m; i++ ){
            int index = sortedResult.at<int>( i, 0 );
            if( D1_score.at( index ) > -10.0 ){
                D1_label.at( index ) = class_label;
            }
        }
    }
}

void VisualWord::writeKmeansResultToDatabase(const Mat &bestlabels)
{
    QSqlQuery query;
    QString SQL_update_command = "UPDATE nyu_depth_patches"
                                 " SET kmeans_class_label = :class_label"
                                 " WHERE id = :id";

    CV_Assert( QSqlDatabase::database().transaction() );
    for( int i = 0; i < bestlabels.rows; i++ ){

        query.prepare( SQL_update_command );
        query.bindValue( ":class_label", int( bestlabels.at<int>( i, 0 ) ) );
        query.bindValue( ":id", i + 1 );
        CV_Assert( query.exec() );
    }
    CV_Assert( QSqlDatabase::database().commit() );

}

void VisualWord::getDataForTrainingFromDatabase( QStringList &D1, QStringList &D2,
                                                 QStringList &N1, QStringList &N2 )
{
    //clear D1, D2, N1, N2
    D1.clear();
    D2.clear();
    N1.clear();
    N2.clear();
    //get the total number of patches in nyu_depth_patches
    QString SQL_query_command = "SELECT COUNT(*) "
                                " FROM nyu_depth_patches";
    QSqlQuery query;
    CV_Assert( query.exec( SQL_query_command ) );
    query.first();
    int nbr_nyu_patches = query.value(0).toInt();
    int nbr_D1 = int( nbr_nyu_patches / 2.0 );

    //get the total number of patches in natural_image_patches
    SQL_query_command = "SELECT COUNT(*) "
                        " FROM natural_image_patches";
    CV_Assert( query.exec( SQL_query_command ) );
    query.first();
    int nbr_natural_image_patches = query.value(0).toInt();
    int nbr_N1 = int( nbr_natural_image_patches / 2.0 );

    //get the nbr_D1 samples from database, whose type is file path
    //indicate where the patch is stored
    SQL_query_command = "SELECT patch_path, patch_name "
                        " FROM nyu_depth_patches "
                        " WHERE id <= :nbr_D1 "
                        " ORDER BY id ;";
    query.prepare( SQL_query_command );
    query.bindValue( ":nbr_D1", nbr_D1 );
    CV_Assert( query.exec());
    QString path, name, full_path;
    while( query.next() ){
        path = query.value("patch_path" ).toString();
        name = query.value("patch_name").toString();
        full_path = path + QDir::separator() + name;
        D1.push_back( full_path );
    }

    //get ( nbr_nyu_patche - nbr_D1 ) sample for D2
    SQL_query_command = "SELECT patch_path, patch_name "
                        " FROM nyu_depth_patches "
                        " WHERE id > :nbr_D1 "
                        " ORDER BY id ;";
    query.prepare( SQL_query_command );
    query.bindValue( ":nbr_D1", nbr_D1 );
    CV_Assert( query.exec());
    while( query.next() ){
        path = query.value("patch_path" ).toString();
        name = query.value("patch_name").toString();
        full_path = path + QDir::separator() + name;
        D2.push_back( full_path );
    }

    //get nbr_N1 samples form N1
    SQL_query_command = "SELECT patch_path, patch_name "
                        " FROM natural_image_patches"
                        " WHERE id <= :nbr_N1 "
                        " ORDER BY id ;";
    query.prepare( SQL_query_command );
    query.bindValue( ":nbr_N1", nbr_N1 );
    CV_Assert( query.exec());
    while( query.next() ){
        path = query.value("patch_path" ).toString();
        name = query.value("patch_name").toString();
        full_path = path + QDir::separator() + name;
        N1.push_back( full_path );
    }

    //get (nbr_natural_image_patches - nbr_N1 ) patches for N2
    SQL_query_command = "SELECT patch_path, patch_name "
                        " FROM natural_image_patches"
                        " WHERE id > :nbr_N1 "
                        " ORDER BY id ;";
    query.prepare( SQL_query_command );
    query.bindValue( ":nbr_N1", nbr_N1 );
    CV_Assert( query.exec());
    while( query.next() ){
        path = query.value("patch_path" ).toString();
        name = query.value("patch_name").toString();
        full_path = path + QDir::separator() + name;
        N2.push_back( full_path );
    }

}

void VisualWord::computeDesriptorMat(const QStringList &D1,
                                     Mat &descriptorMatOfD1 )
{

    imtools::computeHOGDescriptorsMat( descriptorMatOfD1, D1, hog_descr );

}
