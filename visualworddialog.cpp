#include "visualworddialog.h"
#include "ui_visualworddialog.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QSqlQuery>
#include <QDir>
#include <QElapsedTimer>
#include <QProgressDialog>
#include "ui_mainwindow.h"
#include <map>
#include "statistic.h"
#include "depthmakerwithvisualword.h"
#include <stdio.h>

#define FILE_NAME_DESCRIPTOR_NYU_SET1 "HOGData/descriptorMatOfD1.yaml"
#define FILE_NAME_DESCRIPTOR_NATURAL_SET1 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfN1.yaml"
#define FILE_NAME_DESCRIPTOR_NATURAL_SET2 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/HOGData/descriptorMatOfN2.yaml"
#define SVM_CLASSIFIER_PATH "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/svmData/visual_word_classifiers"
#define SVM_CLASSIFIER_PATH_2 "/home/lang/QtProject/build-3DMonster-Desktop_Qt_5_3_GCC_64bit-Debug/svmData/visual_word_classifiers_2"

VisualWordDialog::VisualWordDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::VisualWordDialog)
{
    ui->setupUi(this);
    ui_mainWindow = NULL;
    db = NULL;
    vw = new VisualWord();
    dict = new VisualWordDictionary();
    dict->loadDictionary();
}

VisualWordDialog::~VisualWordDialog()
{
    delete ui;
    delete vw;
    delete dict;
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


void VisualWordDialog::on_trainMultipleSVMClassifierButton_clicked()
{
    if( dict->trainDictionary() ){
        ui->textBrowser->append("SVM training finished!");
    }
    else{
        ui->textBrowser->append("SVM training failed!");
    }

}

void VisualWordDialog::on_prepareDataButton_clicked()
{
}

void VisualWordDialog::on_testClassifierButton_clicked()
{
    HOGDescriptor *hog_descr = new HOGDescriptor( cvSize(80,80), cvSize( 8, 8 ),
                                   cvSize( 8, 8 ), cvSize( 8, 8 ), 9  );
    //load all the SVM classifier

    QDir test_patch_path("/home/lang/dataset/NYUDataset/test_patches");
    QStringList filters;
    filters += "*.png";
    foreach( QString patch_name, test_patch_path.entryList( filters, QDir::Files ) ){
        QString full_path = test_patch_path.path() + QDir::separator() + patch_name;
        Mat test_patch_color = imread( full_path.toLocal8Bit().data() );
        Mat test_patch = imread( full_path.toLocal8Bit().data(), CV_LOAD_IMAGE_GRAYSCALE );

        int best_match_class_id = -1;
        double best_match_score = -10.0;

        best_match_class_id = dict->searchForId( test_patch, best_match_score );

        ui->textBrowser->append("The best score for " + patch_name + " is "
                                + QString::number(best_match_score));
        ui->textBrowser->append("Belongs to class " + QString::number(best_match_class_id ));

        if( best_match_score > -0.5 ){
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

            ui_mainWindow->ImView->setPaintImage( test_patch_color );
            ui_mainWindow->ImView->setPaintImage( depthmap );
        }
    }

}



void VisualWordDialog::on_convertAImageButton_clicked()
{
    if( ui_mainWindow->ImView->isEmpty() )
        return;

    QString filename = QFileDialog::getOpenFileName( this, tr( "Open Image"), SVM_CLASSIFIER_PATH_2,
                                            tr("text Files (*.txt *.yaml)"));

    Mat image = ui_mainWindow->ImView->getCurrentImage();
    MySVM svm;
    svm.load( filename.toLocal8Bit().data() );

    std::vector<Rect> founded_rects;
    std::vector<double> scores;
    VisualWordDictionary::searchImageForGivenWord( image, svm,
                                                       founded_rects, scores );

    for( unsigned i = 0; i < founded_rects.size(); i++ ){
        Rect r = founded_rects[i];
        rectangle( image, r.tl(), r.br(), Scalar( 0, 255, 0 ), 3 );
        printf( "Scores: %f\n", scores[i] );
    }

    ui_mainWindow->ImView->setPaintImage( image );
    ui_mainWindow->statusBar->showMessage( QString::number( founded_rects.size()) +
                                           " rectangles have been found ");

    //DepthMakerWithVisualWord depth_maker;
    //std::vector<Mat> results;
    //depth_maker.generateDepthmap( ui_mainWindow->ImView->getCurrentImage(),
    //                                      results );

    //for( unsigned i = 0; i < results.size(); i++ ){
    //    ui_mainWindow->ImView->setPaintImage( results[i] );
    //}

}

