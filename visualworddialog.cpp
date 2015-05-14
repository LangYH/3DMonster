#include "visualworddialog.h"
#include "ui_visualworddialog.h"
#include <QMessageBox>
#include <QSqlQuery>
#include <QDir>
#include <QElapsedTimer>
#include <QProgressDialog>
#include "ui_mainwindow.h"
#include "patch.h"
#include <map>
#include "statistic.h"
#include "imtools.h"
#include "filters.h"
#include "pyramid.h"

#define FILE_NAME_DESCRIPTOR_NYU_SET1 "HOGData/descriptorMatOfD1.yaml"
#define FILE_NAME_DESCRIPTOR_NATURAL_SET1 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfN1.yaml"
#define FILE_NAME_DESCRIPTOR_NATURAL_SET2 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfN2.yaml"
VisualWordDialog::VisualWordDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::VisualWordDialog)
{
    ui->setupUi(this);
    ui_mainWindow = NULL;
    db = NULL;
    vw = new VisualWord();
}

VisualWordDialog::~VisualWordDialog()
{
    delete ui;
    delete vw;
}

void VisualWordDialog::setDatabase( QSqlDatabase *database )
{
    db = database;
}

void VisualWordDialog::setMainWindowUi(Ui::MainWindow *ui )
{
    ui_mainWindow = ui;
}

void VisualWordDialog::on_loadDataButton_clicked()
{
    if( !db->isOpen() ){
        QMessageBox::information( this, "Error", "Database isn't opened" );
        return;
    }

    ui->textBrowser->append( "Loading data from dataset..." );
    qApp->processEvents();
    vw->loadDataFromDatabase();
    ui->kmeansCentroidsBox->setValue( vw->getDiscoverySamplesSize() / 4 );

    ui->textBrowser->append( "Data loaded!\n" );

}

void VisualWordDialog::on_kmeansClusteringButton_clicked()
{
    ui->textBrowser->append("Deploying kmeans algorithm, please wait..." );
    qApp->processEvents();

    vw->setCentroids( ui->kmeansCentroidsBox->value() );
    vw->setKmeansIterations( ui->kmeansIterationsBox->value() );

    vw->kmeansInitialize();

    ui->textBrowser->append( "training finished");

}

void VisualWordDialog::on_testButton_clicked()
{
    //vw->keepTopResults( 5, 7, STAGE_TWO );

    //QSqlQuery query;
    //QString SQL_update_command = "UPDATE nyu_patches"
    //        " SET kmeans_class_label = :class_label"
    //        " WHERE id = :id";

    //QProgressDialog progress(this);
    //progress.setLabelText( "Saving kmeans data.." );
    //progress.setRange( 0, bestlabels.rows );
    //progress.setModal( true );

    //CV_Assert( QSqlDatabase::database().transaction() );
    //for( int i = 0; i < bestlabels.rows; i++ ){
    //    progress.setValue(i);
    //    qApp->processEvents();

    //    query.prepare( SQL_update_command );
    //    query.bindValue( ":class_label", int( bestlabels.at<int>( i, 0 ) ) );
    //    query.bindValue( ":id", i + 1 );
    //    CV_Assert( query.exec() );
    //}
    //CV_Assert( QSqlDatabase::database().commit() );
    FileStorage fs;
    Mat bestlabels, centers;
    if( fs.open( "kmeans_result.yaml", FileStorage::READ ) ){
        fs["bestlabels"] >> bestlabels;
        fs["centers"] >> centers;
    }
    fs.release();

    vw->writeKmeansResultToDatabase( bestlabels );

    ui->textBrowser->append( "Test Finished!");

}

void VisualWordDialog::on_svmTrainingButton_clicked()
{
    QElapsedTimer timer;
    timer.start();

    if( !vw->isDataLoaded() ){
        ui->textBrowser->append( "Data not load" );
        return;
    }

    int target_class_label = ui->visualWordBox->value();
    int train_iteration = ui->trainingIterationsBox->value();

    ui->textBrowser->append( "Start training class " + QString::number(target_class_label) );
    qApp->processEvents();

    CvSVM svm;
    if( vw->trainOneVisualWord( svm, target_class_label, train_iteration ))
        ui->textBrowser->append( "\nclass " + QString::number(target_class_label)
                                 + " is a visual word.");
    else
        ui->textBrowser->append( "\nclass " + QString::number(target_class_label)
                                 + " is not a visual word.");


    ui->textBrowser->append( "SVM training finished!");
    ui->textBrowser->append( "Elapsed time: " + QString::number( timer.elapsed() / 1000.0 ) +
                             " s\n" );

}

void VisualWordDialog::on_showOneClassButton_clicked()
{
    //double target_svm_score = ui->svmScoreBox->value();

    QString command = "SELECT class_path, depth_average_patch "
            " FROM visual_word "
            " WHERE available = 1 "
            " ORDER BY svm_score DESC "
            " LIMIT 1000; ";

    QSqlQuery query;
    query.prepare( command );
    //query.bindValue(":score", target_svm_score);
    CV_Assert( query.exec( ));

    if( query.size() == 0 )
        return;

    QString path, name, full_path;
    while( query.next() ){
        //read depth patches
        path = query.value(0).toString();
        name = query.value(1).toString();
        full_path= path + QDir::separator() + name;
        Mat depth_patch = imread( full_path.toLocal8Bit().data(), CV_LOAD_IMAGE_GRAYSCALE );
        ui_mainWindow->ImView->setPaintImage( depth_patch );

    }

}

void VisualWordDialog::on_trainAllButton_clicked()
{
    int train_iteration = ui->trainingIterationsBox->value();

    if( !vw->isDataLoaded() ){
        ui->textBrowser->append( "Data not load" );
        return;
    }

    QSqlQuery query;
    QString command = "SELECT kmeans_class_label "
            " FROM nyu_depth_patches "
            " GROUP BY kmeans_class_label "
            " HAVING COUNT(*) > 2 AND kmeans_class_label <> -1 "
            " ORDER BY kmeans_class_label ; ";
    CV_Assert( query.exec( command ) );

    int nbr_class = query.size();
    if( nbr_class == -1 ){
        ui->textBrowser->append( "can't get the size of source image " );
        return;
    }

    QProgressDialog progress(this);
    progress.setRange( 0, nbr_class );
    progress.setModal( true );

    QElapsedTimer timer;
    int target_class_label = 0;
    int class_counts = 0;
    CvSVM svm;
    while( query.next() ){
        timer.restart();
        target_class_label = query.value(0).toInt();
        class_counts++;
        progress.setLabelText( tr( "Training %1 image" ).arg( target_class_label ) );
        progress.setValue( class_counts );
        qApp->processEvents();

        if( vw->trainOneVisualWord( svm, target_class_label, train_iteration ))
            ui->textBrowser->append( "\nclass " + QString::number(target_class_label)
                                     + " is a visual word.");
        else
            ui->textBrowser->append( "\nclass " + QString::number(target_class_label)
                                     + " is not a visual word.");

        ui->textBrowser->append( "SVM training finished!");
        ui->textBrowser->append( "Elapsed time: " + QString::number( timer.elapsed() / 1000.0 ) +
                                 " s\n\n" );
    }


}

void VisualWordDialog::on_findSimilarButton_clicked()
{
    QSqlQuery query;
    QString command = "SELECT class_id, class_path, depth_average_patch, svm_score "
                      " FROM visual_word_2 "
                      " WHERE available = 1 ; ";
    CV_Assert( query.exec( command ) );

    //load data
    std::map<int, Mat> depths;
    std::map<int, double> svm_scores;
    std::map<int,int> tags;
    QString path, name, full_path;
    int class_id;
    while( query.next() ){
        class_id = query.value(0).toInt();
        path = query.value(1).toString();
        name = query.value(2).toString();
        full_path = path + QDir::separator() + name;
        Mat depth = imread( full_path.toLocal8Bit().data(), CV_LOAD_IMAGE_GRAYSCALE );

        depths[class_id] = depth;
        svm_scores[class_id] = query.value(3).toDouble();
        tags[class_id] = 1;
    }

    //find similar pathes
    std::map<int,Mat>::const_iterator map_iter;
    std::map<int,Mat>::const_iterator map_iter2;
    int count = 0;
    std::map<int,Mat>::const_iterator current_iter;
    for( map_iter = depths.begin();map_iter != depths.end();map_iter++ ){
        if( tags[map_iter->first] == 1 ){
            current_iter = map_iter;
            for( map_iter2 = map_iter, ++map_iter2; map_iter2 != depths.end(); map_iter2++ )
                if( tags[map_iter2->first] == 1 ){
                    double s = Statistic::computeCrossCorrelation( current_iter->second,
                                                                   map_iter2->second );
                    if( s < 15.0){
                        if( svm_scores[current_iter->first] >= svm_scores[map_iter2->first] )
                            tags[map_iter2->first] = 0;
                        else{
                            tags[current_iter->first] = 0;
                            current_iter = map_iter2;
                        }
                        count++;
                    }
                }
        }
    }

    //display the available set
    int available_count = 0;
    for( std::map<int,int>::const_iterator iter = tags.begin(); iter != tags.end();
         iter++ ){
        if( iter->second == 1 ){
            ui_mainWindow->ImView->setPaintImage( depths[iter->first] );
            available_count++;
        }
    }


    //updata database
    QSqlDatabase::database().transaction();
    for( std::map<int,int>::const_iterator iter = tags.begin(); iter != tags.end();
         iter++ ){
        if( iter->second == 0 ){
            QString update_command = "UPDATE visual_word_2 "
                                     " SET available = 0 "
                                     " WHERE class_id = :class_id ; ";
            query.prepare( update_command );
            query.bindValue( ":class_id", iter->first);
            assert( query.exec());
        }
    }
    assert( QSqlDatabase::database().commit() );


    ui->textBrowser->append( QString::number(count) + " similar patches have been found!");
    ui->textBrowser->append( QString::number( available_count ) + " available patches have been found!");

}

void VisualWordDialog::on_computeDeviationButton_clicked()
{
    QString command = "SELECT class_path, image_path "
            " FROM visual_word"
            " WHERE class_id = :class_id ;";
    QSqlQuery query;
    query.prepare( command );
    query.bindValue( ":class_id", ui->visualWordBox->value() );
    assert( query.exec() );

    std::vector<Mat> patches;
    if( query.first() ){
        QString path = query.value(0).toString() + QDir::separator() +
                query.value(1).toString();
        QDir target_dir( path );
        QStringList filters;
        filters += "*.png";
        foreach (QString file, target_dir.entryList(filters,QDir::Files)) {
            QString full_path = path + QDir::separator() + file;
            Mat p = imread( full_path.toLocal8Bit().data(), CV_LOAD_IMAGE_GRAYSCALE );
            patches.push_back(p);
        }
    }
    double dev = Statistic::computeDeviationOfMatrixes( patches );

    ui->textBrowser->append( "Deviation:" + QString::number(dev) );

}


static void prepareNegativeSamplesForMultipleSVMTraining( Mat &negativeDescriptorMat )
{

    //Mat neg1, neg2;
    //FileStorage fs;
    //if( fs.open( FILE_NAME_DESCRIPTOR_NATURAL_SET1, FileStorage::READ ) ){
    //    fs["N1"] >> neg1;
    //}
    //fs.release();

    //negativeDescriptorMat = neg1.rowRange(0,50).clone();

    //if( fs.open( FILE_NAME_DESCRIPTOR_NATURAL_SET2, FileStorage::READ ) ){
    //    fs["N2"] >> negativeDescriptorMat;
    //}
    //fs.release();
    //vconcat( neg1, neg2, negativeDescriptorMat );

    QString command = "SELECT class_path, image_path "
            " FROM visual_word"
            " WHERE available = 1 AND svm_score <= 7.5 "
            " ORDER BY class_id ;";
    QSqlQuery query;
    assert( query.exec(command) );

    HOGDescriptor *hog_descr = new HOGDescriptor( cvSize(80,80), cvSize( 8, 8 ),
                                   cvSize( 8, 8 ), cvSize( 8, 8 ), 9  );

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

void VisualWordDialog::on_trainMultipleSVMClassifierButton_clicked()
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


    HOGDescriptor *hog_descr = new HOGDescriptor( cvSize(80,80), cvSize( 8, 8 ),
                                   cvSize( 8, 8 ), cvSize( 8, 8 ), 9  );

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

    ui->textBrowser->append("SVM training finished!");

}

void VisualWordDialog::on_prepareDataButton_clicked()
{
}

void VisualWordDialog::on_testClassifierButton_clicked()
{
    HOGDescriptor *hog_descr = new HOGDescriptor( cvSize(80,80), cvSize( 8, 8 ),
                                   cvSize( 8, 8 ), cvSize( 8, 8 ), 9  );
    //load all the SVM classifier
    std::map<int, CvSVM*> classifiers;
    VisualWord::loadAllSVMClassifiers( classifiers );

    QDir test_patch_path("/home/lang/dataset/NYUDataset/test_patches");
    QStringList filters;
    filters += "*.png";
    foreach( QString patch_name, test_patch_path.entryList( filters, QDir::Files ) ){
        QString full_path = test_patch_path.path() + QDir::separator() + patch_name;
        Mat test_patch_color = imread( full_path.toLocal8Bit().data() );
        Mat test_patch = imread( full_path.toLocal8Bit().data(), CV_LOAD_IMAGE_GRAYSCALE );

        int best_match_class_id = -1;
        double best_match_score = -10.0;
        std::map<int, CvSVM*>::const_iterator iter;
        for( iter = classifiers.begin(); iter != classifiers.end(); iter++ ){
            std::vector<float> descr;
            hog_descr->compute( test_patch, descr, Size(0, 0 ), Size( 0, 0 ) );
            float score = -( iter->second->predict( Mat(descr).t(), true ) );
            if( score > best_match_score ){
                best_match_score = score;
                best_match_class_id = iter->first;
            }
        }

        ui->textBrowser->append("The best score for " + patch_name + " is "
                                + QString::number(best_match_score));
        ui->textBrowser->append("Belongs to class " + QString::number(best_match_class_id ));

        if( best_match_score > -1.0 ){
            //get the corresponding depthmap
            QString command = "SELECT class_path, depth_average_patch "
                    " FROM visual_word "
                    " WHERE class_id = :class_id ;";
            QSqlQuery query;
            query.prepare(command);
            query.bindValue(":class_id", best_match_class_id );
            CV_Assert( query.exec() );

            if( query.size() == 0 )
                break;
            query.first();
            QString depthmap_path = query.value(0).toString() + QDir::separator() +
                    query.value(1).toString();
            Mat depthmap = imread( depthmap_path.toLocal8Bit().data() );

            ui_mainWindow->ImView->setPaintImage( test_patch_color );
            ui_mainWindow->ImView->setPaintImage( depthmap );
        }
    }

    VisualWord::cleanAllSVMClassifiers( classifiers );
}

static void initialDepthMap( std::vector<Mat> const &images, std::vector<Mat> &depths )
{
    for( unsigned i = 0; i < images.size(); i++ ){
        Mat depth = images[i].clone();
        depths.push_back( depth );
    }
}

void VisualWordDialog::on_convertAImageButton_clicked()
{
    if( ui_mainWindow->ImView->isEmpty() )
        return;


    Mat image = ui_mainWindow->ImView->getCurrentImage();


    std::vector<Mat> pyrs;
    std::vector< std::vector<Mat> > patches_array;
    std::vector< std::vector<Point> > coordinates_array;
    std::vector< std::vector<PATCH_TYPE> > overlappedPatchSymbols_array;
    std::vector< std::vector<PATCH_TYPE> > flatPatchSymbols_array;

    samplePatchesForDepthGeneration( image, patches_array, coordinates_array );

    Patch *patchExtracter;
    patchExtracter = new Patch( 80 );


    //drop the first two layers of pyramid
    Pyramid imPyramid( 3, 2, 1.0 );
    imPyramid.buildGaussianPyramid( image, pyrs );
    pyrs.erase( pyrs.begin() );
    pyrs.erase( pyrs.begin() );

    std::vector<Mat> patches_in_single_image;
    std::vector<Point> coordinates_in_single_image;
    std::vector<int> number_vector;
    number_vector.push_back( 20 );
    number_vector.push_back( 20 );
    number_vector.push_back( 10 );
    number_vector.push_back( 10 );
    for( unsigned int i = 0; i < pyrs.size(); i++ ){
        std::vector<Mat>().swap( patches_in_single_image );
        std::vector<Point>().swap( coordinates_in_single_image );
        patchExtracter->randomSamplePatches( pyrs[i], patches_in_single_image,
                             coordinates_in_single_image, number_vector[i] );
        patches_array.push_back( patches_in_single_image );
        coordinates_array.push_back( coordinates_in_single_image );
    }

    //intialize overlap symbols and flatPatchSymbols
    overlappedPatchSymbols_array.assign( patches_array.size(), std::vector<PATCH_TYPE>() );
    flatPatchSymbols_array.assign( patches_array.size(), std::vector<PATCH_TYPE>() );
    for( unsigned int i = 0; i < patches_array.size(); i++ ){
        overlappedPatchSymbols_array[i].assign( patches_array[i].size(), POSITIVE );
        flatPatchSymbols_array[i].assign( patches_array[i].size(), POSITIVE );
    }

    //remove overlapped and flat patches
    patchExtracter->detectOverlappedPatchesInPyramid( patches_array,
                                                      overlappedPatchSymbols_array,
                                                      60,
                                                      CROSS_CORRELATION );

    patchExtracter->detectFlatPatchesInPyramid( patches_array,
                                                flatPatchSymbols_array,
                                                10.0,
                                                DEVIATION );

    HOGDescriptor *hog_descr = new HOGDescriptor( cvSize(80,80), cvSize( 8, 8 ),
                                                  cvSize( 8, 8 ), cvSize( 8, 8 ), 9  );
    //load all the SVM classifier
    std::map<int, CvSVM*> classifiers;
    VisualWord::loadAllSVMClassifiers( classifiers );

    std::vector<Mat> depths;
    initialDepthMap( pyrs, depths );
    for( unsigned int i = 0; i < patches_array.size(); i++ ){
        for(  unsigned int j = 0; j < patches_array[i].size(); j++ ){

            if( overlappedPatchSymbols_array[i][j] == POSITIVE &&
                            flatPatchSymbols_array[i][j] == POSITIVE ){
                int best_match_class_id = -1;
                double best_match_score = -10.0;
                std::map<int, CvSVM*>::const_iterator iter;
                for( iter = classifiers.begin(); iter != classifiers.end(); iter++ ){
                    std::vector<float> descr;
                    hog_descr->compute( patches_array[i][j], descr, Size(0, 0 ), Size( 0, 0 ) );
                    float score = -( iter->second->predict( Mat(descr).t(), true ) );
                    if( score > best_match_score ){
                        best_match_score = score;
                        best_match_class_id = iter->first;
                    }
                }

                if( best_match_score > -1.0 ){
                    //get the corresponding depthmap
                    QString command = "SELECT class_path, depth_average_patch "
                                      " FROM visual_word_2 "
                                      " WHERE class_id = :class_id ;";
                    QSqlQuery query;
                    query.prepare(command);
                    query.bindValue(":class_id", best_match_class_id );
                    CV_Assert( query.exec() );

                    if( query.size() == 0 )
                        break;
                    query.first();
                    QString depthmap_path = query.value(0).toString() + QDir::separator() +
                                    query.value(1).toString();
                    Mat depthmap = imread( depthmap_path.toLocal8Bit().data() );
                    Filters filters;
                    filters.guidedFilter( depthmap, patches_array[i][j], depthmap, 30, 0.1 );
                    normalize( depthmap, depthmap, 0, 1.0, NORM_MINMAX, CV_32FC1 );
                    double ratio = ( double( depths[i].rows ) - coordinates_array[i][j].y ) /
                            depths[i].rows * 255.0;
                    Mat temp = depthmap * ratio;
                    temp.convertTo( depthmap, CV_8UC1 );

                    if( depths[i].channels() == 3 && depthmap.channels() == 1 )
                        cvtColor( depthmap, depthmap, CV_GRAY2BGR );

                    Mat imageROI = depths[i]( cv::Rect( coordinates_array[i][j].x,
                                                         coordinates_array[i][j].y,
                                                         depthmap.cols, depthmap.rows ) );
                    cv::addWeighted( imageROI, 0., depthmap, 1.0, 0., imageROI );

                }
            }

        }
    }

    for( unsigned i = 0; i < depths.size(); i++ ){
        Mat elarge_image;
        cv::resize( depths[i], elarge_image, image.size() );
        ui_mainWindow->ImView->setPaintImage( elarge_image );
    }
    delete patchExtracter;
}

void samplePatchesForDepthGeneration( const Mat &image,
                                      std::vector<Mat> patches,
                                      std::vector<Point> coordinates )
{

}
