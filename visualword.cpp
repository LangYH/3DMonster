#include "visualword.h"
#include <QSqlQuery>
#include <QVariant>
#include <QDir>
#include <sys/time.h>
#include "imtools.h"
#include <QFile>

#define NUMBER_OF_DISCOVERY_SET1 118242
#define CROSS_VALIDATION_THRESHOLD 3
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
    svm_params.term_crit = TermCriteria( CV_TERMCRIT_ITER, 48, 1e-6 );

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

void VisualWord:: visualWordsTrainingWithCrossValidation()
{

}

bool VisualWord::trainOneVisualWord( CvSVM &svm, const int init_class_label, const int iteration )
{
    int iter = 0;
    initDatabaseClassLabel();
    while( iter < iteration ){
        //training step
        cleanDatabaseClassLabel( STAGE_ONE );
        svmTrain( svm, init_class_label, STAGE_ONE );
        svmDetect( svm, init_class_label, STAGE_ONE );
        if( !keepTopResults( 5, init_class_label, STAGE_ONE ) ){
            return false;
        }

        //cross validation step
        cleanDatabaseClassLabel( STAGE_TWO );
        svmTrain( svm, init_class_label, STAGE_TWO );
        svmDetect( svm, init_class_label, STAGE_TWO );
        if( !keepTopResults( 5, init_class_label, STAGE_TWO)){
            return false;
        }

        iter++;

    }

    return true;

}

void VisualWord::svmTrain( CvSVM &svm, const int class_label, CROSS_VALIDATION_SYMBOL cv_symbol)
{
    Mat negativeDescriptorMat;
    QString SELECT_POSITIVE_SAMPLE_COMMAND;

    FileStorage fs;
    QSqlQuery query;
    if( cv_symbol == STAGE_ONE ){
        SELECT_POSITIVE_SAMPLE_COMMAND =  "SELECT patch_path, patch_name "
                                          " FROM nyu_patches "
                                          " WHERE id <= :dsize AND test_label = :class_label ;";

        if( fs.open( FILE_NAME_DESCRIPTOR_NATURAL_SET1, FileStorage::READ ) ){
            fs["N1"] >> negativeDescriptorMat;
        }

    } else {
        SELECT_POSITIVE_SAMPLE_COMMAND =  "SELECT patch_path, patch_name "
                                          " FROM nyu_patches "
                                          " WHERE id > :dsize AND test_label = :class_label ;";

        if( fs.open( FILE_NAME_DESCRIPTOR_NATURAL_SET2, FileStorage::READ ) ){
            fs["N2"] >> negativeDescriptorMat;
        }


    }

    //get postitve samples' information
    query.prepare( SELECT_POSITIVE_SAMPLE_COMMAND );
    query.bindValue( ":dsize", NUMBER_OF_DISCOVERY_SET1 );
    query.bindValue( ":class_label", class_label );
    CV_Assert( query.exec( ) );

    QStringList positive_list;
    while( query.next() ){
        QString path = query.value(0).toString();
        QString name = query.value(1).toString();
        QString full_path = path + QDir::separator() + name;
        positive_list.push_back( full_path );
    }
    //*********************************************************

    Mat positive_descriptor_matrix;
    computeDesriptorMat( positive_list, positive_descriptor_matrix );

    Mat training_data_Mat;
    vconcat( positive_descriptor_matrix, negativeDescriptorMat, training_data_Mat );

    Mat labels_Mat( positive_descriptor_matrix.rows + negativeDescriptorMat.rows,
                    1, CV_32FC1, Scalar(-1.0) );

    labels_Mat( cv::Rect( 0, 0, 1, positive_descriptor_matrix.rows ) ).setTo( Scalar(1.0) );

    svm.train( training_data_Mat, labels_Mat, Mat(), Mat(), svm_params );

}

void VisualWord::svmDetect(CvSVM &svm, const int class_label, CROSS_VALIDATION_SYMBOL cv_symbol)
{
    //cross validation step
    QSqlQuery query;
    QString UPDATE_COMMAND = "UPDATE nyu_patches "
                             " SET test_label = :label, test_score = :score "
                             " WHERE id = :id";

    if( cv_symbol == STAGE_ONE ){
        for( int i = 0; i < D2.size(); i++ ){

            int id_in_DB = NUMBER_OF_DISCOVERY_SET1 + i + 1;

            Mat im = imread( D2[i].toLocal8Bit().data(),
                             CV_LOAD_IMAGE_GRAYSCALE );


            //svm predict process
            std::vector<float> descr;
            hog_descr->compute( im, descr, Size(0, 0 ), Size( 0, 0 ) );
            float score = svm.predict( Mat(descr).t(), true );
            float response = svm.predict( Mat(descr).t() );

            //update database
            if( (-score) > -1.0 ){
                query.prepare( UPDATE_COMMAND );
                query.bindValue( ":label", class_label );
                query.bindValue( ":score", -score );
                query.bindValue( ":id", id_in_DB );
                CV_Assert( query.exec() );
            }

        }
    }
    else{
        for( int i = 0; i < D1.size(); i++ ){

            int id_in_DB = i + 1;

            Mat im = imread( D1[i].toLocal8Bit().data(),
                             CV_LOAD_IMAGE_GRAYSCALE );


            //svm predict process
            std::vector<float> descr;
            hog_descr->compute( im, descr, Size(0, 0 ), Size( 0, 0 ) );
            float score = svm.predict( Mat(descr).t(), true );
            float response = svm.predict( Mat(descr).t() );

            //update database
            if( (-score) > -1.0 ){
                query.prepare( UPDATE_COMMAND );
                query.bindValue( ":label", class_label );
                query.bindValue( ":score", -score );
                query.bindValue( ":id", id_in_DB );
                CV_Assert( query.exec() );
            }

        }


    }

}

void VisualWord::getDataForTrainingFromDatabase( QStringList &D1, QStringList &D2,
                                                 QStringList &N1, QStringList &N2 )
{
    //clear D1, D2, N1, N2
    D1.clear();
    D2.clear();
    N1.clear();
    N2.clear();
    //get the total number of patches in nyu_patches
    QString SQL_query_command = "SELECT COUNT(*) "
                                " FROM nyu_patches";
    QSqlQuery query;
    query.exec( SQL_query_command );
    query.first();
    int nbr_nyu_patches = query.value(0).toInt();
    int nbr_D1 = int( nbr_nyu_patches / 2.0 );

    //get the total number of patches in natural_image_patches
    SQL_query_command = "SELECT COUNT(*) "
                        " FROM natural_image_patches";
    query.exec( SQL_query_command );
    query.first();
    int nbr_natural_image_patches = query.value(0).toInt();
    int nbr_N1 = int( nbr_natural_image_patches / 2.0 );

    //get the nbr_D1 samples from database, whose type is file path
    //indicate where the patch is stored
    SQL_query_command = "SELECT patch_path, patch_name "
                        " FROM nyu_patches "
                        " WHERE id <= " + QString::number( nbr_D1 );
    query.exec( SQL_query_command );
    QString path, name, full_path;
    while( query.next() ){
        path = query.value("patch_path" ).toString();
        name = query.value("patch_name").toString();
        full_path = path + QDir::separator() + name;
        D1.push_back( full_path );
    }

    //get ( nbr_nyu_patche - nbr_D1 ) sample for D2
    SQL_query_command = "SELECT patch_path, patch_name "
                        " FROM nyu_patches "
                        " WHERE id > " + QString::number( nbr_D1 );
    query.exec( SQL_query_command );
    while( query.next() ){
        path = query.value("patch_path" ).toString();
        name = query.value("patch_name").toString();
        full_path = path + QDir::separator() + name;
        D2.push_back( full_path );
    }

    //get nbr_N1 samples form N1
    SQL_query_command = "SELECT patch_path, patch_name "
                        " FROM natural_image_patches"
                        " WHERE id <= " + QString::number( nbr_N1 );
    query.exec( SQL_query_command );
    while( query.next() ){
        path = query.value("patch_path" ).toString();
        name = query.value("patch_name").toString();
        full_path = path + QDir::separator() + name;
        N1.push_back( full_path );
    }

    //get (nbr_natural_image_patches - nbr_N1 ) patches for N2
    SQL_query_command = "SELECT patch_path, patch_name "
                        " FROM natural_image_patches"
                        " WHERE id > " + QString::number( nbr_N1 );
    query.exec( SQL_query_command );
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

void VisualWord::initDatabaseClassLabel()
{
    QSqlQuery query;
    QString reset_command = "UPDATE nyu_patches "
                            " SET test_label = kmeans_class_label; ";
    CV_Assert( query.exec(reset_command) );
}

void VisualWord::cleanDatabaseClassLabel( CROSS_VALIDATION_SYMBOL cv_symbol)
{
    QSqlQuery query;
    if( cv_symbol == STAGE_ONE ){
        QString clean_database_command = "UPDATE nyu_patches "
                             " SET test_label = -1, test_score = 0.0 "
                             " WHERE id > :dsize ;";
        query.prepare( clean_database_command );
        query.bindValue( ":dsize", NUMBER_OF_DISCOVERY_SET1 );
        CV_Assert( query.exec() );
    }
    else if( cv_symbol == STAGE_TWO ){
        QString clean_database_command = "UPDATE nyu_patches "
                             " SET test_label = -1, test_score = 0.0 "
                             " WHERE id <= :dsize ;";
        query.prepare( clean_database_command );
        query.bindValue( ":dsize", NUMBER_OF_DISCOVERY_SET1 );
        CV_Assert( query.exec() );
    }
}

bool VisualWord::keepTopResults( const int m, const int class_label, CROSS_VALIDATION_SYMBOL cv_symbol)
{
    QSqlQuery query;
    QString select_command, update_command;
    if( cv_symbol == STAGE_ONE ){
        select_command = "SELECT test_score "
                         " FROM nyu_patches"
                         " WHERE id > :dsize AND test_label = :label "
                         " ORDER BY ABS(test_score) DESC; ";
        update_command = "UPDATE nyu_patches "
                         " SET test_label = -1, test_score = 0.0 "
                         " WHERE id > :dsize AND test_label = :label AND ABS( test_score ) < :threshold ;";

    }
    else{
        select_command = "SELECT test_score "
                         " FROM nyu_patches"
                         " WHERE id <= :dsize AND test_label = :label "
                         " ORDER BY ABS(test_score) DESC; ";
        update_command = "UPDATE nyu_patches "
                         " SET test_label = -1, test_score = 0.0 "
                         " WHERE id <= :dsize AND test_label = :label AND ABS( test_score ) < :threshold ;";

    }

    query.prepare( select_command );
    query.bindValue( ":dsize", NUMBER_OF_DISCOVERY_SET1 );
    query.bindValue( ":label", class_label );
    CV_Assert( query.exec() );

    //for those whose svm score is less than the top m patches,
    //set their label and score to default
    float score_threshold;
    int count = 0;
    while( query.next() && count < m ){
        score_threshold = query.value(0).toFloat();
        count++;
    }

    if( count < CROSS_VALIDATION_THRESHOLD ){
        return false;
    }
    else{
        query.prepare( update_command );
        query.bindValue( ":dsize", NUMBER_OF_DISCOVERY_SET1 );
        query.bindValue( ":label", class_label );
        query.bindValue( ":threshold", score_threshold );
        CV_Assert( query.exec() );
        return  true;
    }

}

void VisualWord::writeKmeansResultToDatabase(const Mat &bestlabels)
{
    QSqlQuery query;
    QString SQL_update_command = "UPDATE nyu_patches"
                                 " SET class_label = :class_label"
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
