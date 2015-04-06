#include "visualworddialog.h"
#include "ui_visualworddialog.h"
#include <QMessageBox>
#include <QSqlQuery>
#include <QDir>
#include <QElapsedTimer>
#include <QProgressDialog>
#include "ui_mainwindow.h"
#include "patch.h"

#define FILE_NAME_DESCRIPTOR_NYU_SET1 "HOGData/descriptorMatOfD1.yaml"

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
    int target_class_label = ui->visualWordBox->value();

    QString command = "SELECT patch_path, patch_name, source_image_path, source_image_name, coordinate_x, coordinate_y, layer "
            " FROM nyu_depth_patches "
            " WHERE test_label = :label " //AND id <= 46848"
            " ORDER BY test_score ;";

    QSqlQuery query;
    query.prepare( command );
    query.bindValue(":label", target_class_label);
    CV_Assert( query.exec( ));

    if( query.size() == 0 )
        return;

    QString path, name, full_path;
    QString source_image_path, source_image_name, source_image_full_path;
    int coordinate_x, coordinate_y, layer;
    std::vector<Mat> mat_list;
    std::vector<Mat> patches_list;
    while( query.next() ){
        //read depth patches
        path = query.value(0).toString();
        name = query.value(1).toString();
        full_path= path + QDir::separator() + name;
        Mat depth_patch = imread( full_path.toLocal8Bit().data(), CV_LOAD_IMAGE_GRAYSCALE );
        ui_mainWindow->ImView->setPaintImage( depth_patch );

        //read image pathces
        source_image_path = query.value(2).toString();
        source_image_name = query.value(3).toString();
        coordinate_x = query.value(4).toInt();
        coordinate_y = query.value(5).toInt();
        layer = query.value(6).toInt();
        source_image_full_path = source_image_path + QDir::separator() + source_image_name;
        Mat source_image = imread( source_image_full_path.toLocal8Bit().data() );
        Mat image_patch;
        Patch::getPatchForGivenCoordinates( source_image, image_patch, 3, 2, 1.6,
                                            Point( coordinate_x, coordinate_y ), layer );

        ui_mainWindow->ImView->setPaintImage( image_patch );
        depth_patch.convertTo( depth_patch, CV_32FC1 );
        image_patch.convertTo( image_patch, CV_32FC1 );
        mat_list.push_back( depth_patch );
        patches_list.push_back( image_patch );
    }

    Mat average( mat_list[0].rows, mat_list[0].cols, CV_32FC1, Scalar(0) );
    for( unsigned i = 0; i < mat_list.size(); i++ ){
        average += mat_list.at(i);
    }

    cv::normalize( average, average, 0, 255, NORM_MINMAX, CV_8UC1 );

    //ui_mainWindow->ImView->setPaintImage( average );


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
