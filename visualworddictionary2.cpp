#include "visualworddictionary2.h"
#include <iostream>
#include <QSqlQuery>
#include <QElapsedTimer>
#include <QVariant>
#include <assert.h>
#include <QDir>
#include "imtools.h"
#include <algorithm>
#include <QRegularExpression>
#include "databasemanager.h"
#include <queue>
#include <assert.h>
#include "visualword2.h"
#include "statistic.h"

#define FILE_NAME_DESCRIPTOR_NATURAL_SET1 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfN1.yaml"
#define FILE_NAME_DESCRIPTOR_NATURAL_SET2 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfN2.yaml"
#define CLUSTER_STORING_PATH "/home/lang/dataset/NYUDataset/visual_word_with_hard_negative_mining"
#define VISUAL_WORD_DETECTOR_PATH "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/svmData/visual_word_detectors/"

VisualWordDictionary2::VisualWordDictionary2()
{
    hog_descr = new HOGDescriptor( cvSize(80,80), cvSize( 8, 8 ),
                                   cvSize( 8, 8 ), cvSize( 8, 8 ), 9  );
    DatabaseManager::loadVisualWordTrainingData( I1, I2, D1,D2,N1,N2);
}

void VisualWordDictionary2::parallelTrainVisualWordDetector()
{
    trainVisualWordInitialization();
    assert( visual_word_IDs.size() != 0 );
    QString save_svm_path = VISUAL_WORD_DETECTOR_PATH;

#pragma omp parallel
    {
#pragma omp for schedule( dynamic, 1 ) nowait
        for( unsigned i = 0; i < visual_word_IDs.size();i++){
            MySVM svm;
            trainVisualWordDetector( visual_word_IDs[i], 3, svm );
            QString svm_save_path = save_svm_path + "svm_detector_"
                            + QString::number(visual_word_IDs[i]) + ".txt";
            svm.save(svm_save_path.toLocal8Bit().data());
        }
    }
}

void VisualWordDictionary2::trainVisualWordInitialization(bool clean_overlaps,
                                                          double threshold)
{
    //if clean_overlaps == true, then clean the overlapped cluster for
    //given threshold
    if( clean_overlaps == true ){
        cleanOverlapClusters( threshold );
    }

    //get all available visual word IDs from database
    QSqlQuery query;
    QString command = "SELECT class_id"
                      " FROM visual_word_3"
                      " WHERE available = 1 "
                      " ORDER BY class_id";
    CV_Assert( query.exec( command ) );

    while( query.next() ){
        visual_word_IDs.push_back( query.value("class_id").toInt() );
    }

    //get the corresponding patches cluster for each visual word
    QString data_command = "SELECT class_path, image_path "
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

void VisualWordDictionary2::computePersonSimMatrix( std::vector<int> depths_IDs,
                                                    std::map<int,Mat> depths,
                                                    Mat& person_sim_matrix )
{
    int nr = depths_IDs.size();
    int nc = depths_IDs.size();

    person_sim_matrix.create( nr, nc, CV_64FC1 );

    for( int i = 0; i < nr; i++ ){
        double *data = person_sim_matrix.ptr<double>(i);
        data = data + i;
        for( int j = i; j < nc; j++ ){
            *data++ = Statistic::computePersonSimlarity( depths[depths_IDs[i]],
                    depths[depths_IDs[j]] );
        }
    }
}

void VisualWordDictionary2::cleanOverlapClusters( double threshold )
{
    QSqlQuery query;
    QString reset_command = "UPDATE visual_word_3 "
                      " SET available = 1 ";
    assert( query.exec( reset_command ) );

    //get Data From database

    std::vector<int> depths_IDs;
    std::map<int, Mat> depths;
    std::map<int, bool> visual_word_permissions;

    QString command = "SELECT class_id"
                      " FROM visual_word_3"
                      " ORDER BY class_id";
    CV_Assert( query.exec( command ) );

    while( query.next() ){
        depths_IDs.push_back( query.value("class_id").toInt() );
    }

    //get the corresponding patches cluster for each visual word
    QString data_command = "SELECT class_path "
                           " FROM visual_word_3"
                           " WHERE class_id= :class_id;";
    for( unsigned i = 0; i < depths_IDs.size(); i++ ){
        int current_class_label = depths_IDs[i];
        query.prepare( data_command );
        query.bindValue(":class_id", current_class_label );
        assert( query.exec() );

        assert( query.first() );

        QString class_path = query.value("class_path").toString();
        QString average_depth_path= class_path + QDir::separator() + "average_depth.png";

        if( depths.count( current_class_label ) == 0 ){
            visual_word_permissions[current_class_label] = true;
            depths[current_class_label] =
                            imread( average_depth_path.toLocal8Bit().data(), CV_LOAD_IMAGE_GRAYSCALE );
        }
    }

    Mat person_sim_matrix;
    computePersonSimMatrix( depths_IDs, depths, person_sim_matrix  );

    QString test_path = "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/svmData/test";
    QDir test_dir(test_path);

    int nr = depths_IDs.size();
    int nc = depths_IDs.size();
    for( int i = 0; i < nr; i++ ){
        //if the i cluster has been detected as an ovelapped one, then jump over
        if( visual_word_permissions[depths_IDs[i]] == false )
            continue;

        QString subDir = "cluster_" + QString::number( depths_IDs[i]  );
        assert( test_dir.mkdir( subDir ));
        QString save_path = test_path + QDir::separator()
                        + subDir + QDir::separator() + "cluster_"
                        + QString::number( depths_IDs[i]) + ".png";
        Mat d = depths[depths_IDs[i]];
        imwrite( save_path.toLocal8Bit().data(), d );

        const double *data = person_sim_matrix.ptr<double>(i);
        data = data + i + 1;
        for( int j = i + 1; j < nc; j++ ){
            if( *data++ > threshold ){
                visual_word_permissions[depths_IDs[j]] = false;

                QString save_path_other = test_path + QDir::separator()
                                + subDir + QDir::separator() + "cluster_"
                                + QString::number(depths_IDs[j]) + ".png";
                Mat depth = depths[depths_IDs[j]];
                imwrite( save_path_other.toLocal8Bit().data(), depth );
            }
        }
    }

    QString update_command = "UPDATE visual_word_3 "
                             " SET available = 0 "
                             " WHERE class_id = :class_id ; ";
    QSqlDatabase::database().transaction();
    for( int i = 0; i < nc; i++ ){
        if( visual_word_permissions[ depths_IDs[i]] == false ){
            query.prepare( update_command );
            query.bindValue( ":class_id", depths_IDs[i] );
            assert(query.exec());
        }
    }
    assert( QSqlDatabase::database().commit() );
}

void VisualWordDictionary2::trainVisualWordDetector( int class_ID,
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
    getNegativeDescriptorTraining( negative_descriptor_Mat );

    VisualWord2::svmTrain( positive_descriptor_Mat, negative_descriptor_Mat, svm );

    //hard examples mining
    for( int i = 0; i < iteration; i++ ){
        std::vector<Mat> hard_examples;
        VisualWord2::getHardExmaples( svm, negative_image_list, hard_examples );

        if( hard_examples.size() > 0 ){
            VisualWord2::trainOneSVMDetectorWithHardExamples( positive_descriptor_Mat, negative_descriptor_Mat,
                                                 hard_examples, hog_descr, svm );
        }

        std::vector<Mat>().swap( hard_examples );
    }

}

void VisualWordDictionary2::getNegativeDescriptorTraining( Mat &negative_descriptor_Mat )
{
    negative_descriptor_Mat = negative_descriptor_1;
}

void VisualWordDictionary2::trainOneSVMDetectorWithHardExamples( const Mat &positive_descriptor_Mat,
                                          const Mat &negative_descriptor_Mat,
                                          const std::vector<Mat> &hard_examples,
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
