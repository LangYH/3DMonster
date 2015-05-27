#include "visualworddictionary.h"
#include <QDir>
#include <QSqlQuery>
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

bool VisualWordDictionary::trainDictionary()
{
    return trainMultipleSVMClassifier();
}

bool VisualWordDictionary::loadDictionary()
{
    loadAllSVMClassifiers( classifiers );
    return true;
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
    //prepare negative samples, which are from natural image set
    Mat negative_descriptor_mat;
    prepareNegativeSamplesForMultipleSVMTraining( negative_descriptor_mat );

    //SVM parameter setting
    CvMat *weights = cvCreateMat( 2, 1, CV_32FC1 );
    cvmSet( weights, 0, 0, 1.0 );
    cvmSet( weights, 1, 0, 0.05 );
    CvSVMParams svm_params;
    svm_params.svm_type = SVM::C_SVC;
    svm_params.C = 1.0;
    svm_params.class_weights = weights;
    svm_params.kernel_type = SVM::LINEAR;
    svm_params.term_crit = TermCriteria( CV_TERMCRIT_ITER, 30, 1e-6 );


    //train a SVM classifier for every visual word whose svm_score in depth space
    //is bigger than 7.5
    QString command = "SELECT class_id, class_path, image_path "
            " FROM visual_word_2"
            " WHERE available = 1 AND svm_score > 7.5 "
            " ORDER BY class_id ;";
    QSqlQuery query;
    query.prepare( command );
    assert( query.exec() );

    std::map<int,Mat> positive_samples;
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
        vconcat( negative_descriptor_mat, other_class_mat, negative_big_mat );

        Mat positive_labels_mat( iter->second.rows, 1, CV_32FC1, Scalar(1.0) );
        Mat negative_labels_mat( negative_big_mat.rows, 1, CV_32FC1, Scalar(-1.0));

        //now we can have the training examples and labels for SVM trainging
        Mat samples_matrixes, labels_matrixes;
        vconcat( iter->second, negative_big_mat, samples_matrixes );
        vconcat( positive_labels_mat, negative_labels_mat, labels_matrixes );

        //svm training...
        QString classifier_save_path = "svmData/visual_word_classifiers_2/visual_word_"
                        + QString::number( iter->first ) + ".txt";
        CvSVM svm;
        svm.train( samples_matrixes, labels_matrixes, Mat(), Mat(), svm_params );
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
        svm.save( classifier_save_path.toLocal8Bit().data() );
    }

    return true;

}

void VisualWordDictionary::cleanAllSVMClassifiers(std::map<int, CvSVM *> &classifiers)
{
    std::map<int, CvSVM*>::const_iterator iter;
    for( iter = classifiers.begin(); iter != classifiers.end(); iter++ ){
        delete iter->second;
    }
}

void VisualWordDictionary::loadAllSVMClassifiers(std::map<int, CvSVM *> &classifiers)
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
            classifiers[class_id] = new CvSVM;
            classifiers[class_id]->load( full_path.toLocal8Bit().data() );
        }
    }

}
