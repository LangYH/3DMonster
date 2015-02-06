#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QFileInfo>
#include "connectdatabasedialog.h"
#include "ui_connectdatabasedialog.h"
#include "crossbilateralfilterdialog.h"
#include "ui_crossbilateralfilterdialog.h"
#include "guidedfilterdialog.h"
#include "ui_guidedfilterdialog.h"
#include "databasemanager.h"
#include "filters.h"
#include <QElapsedTimer>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    lastPath = "/home/lang/Pictures/";

    //bind signals to slots
    connect( ui->actionE_xit, SIGNAL(triggered()), this, SLOT(close()) );
    connect( ui->action_Remove_image, SIGNAL(triggered()), ui->ImView, SLOT(eraseTopImage()) );

    //set context menu
    ui->ImView->addAction( ui->action_Remove_image );
    ui->ImView->setContextMenuPolicy( Qt::ActionsContextMenu );

    statusBarMessageLabel = new QLabel("Open image to start processing" );
    QFont statusBarFont("Times", 12, QFont::Bold );
    statusBarMessageLabel->setFont( statusBarFont );
    statusBarMessageLabel->setMinimumSize( statusBarMessageLabel->sizeHint() );
    statusBarMessageLabel->setAlignment( Qt::AlignLeft );
    statusBarMessageLabel->setIndent( 3 );

    copyRightInfoLabel = new QLabel( "Copy right: LangYH");
    copyRightInfoLabel ->setMinimumSize( copyRightInfoLabel ->sizeHint() );
    copyRightInfoLabel ->setAlignment( Qt::AlignRight );
    copyRightInfoLabel ->setIndent( 3 );

    statusBar()->addWidget( statusBarMessageLabel, 1 );
    statusBar()->addWidget( copyRightInfoLabel );
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_action_Open_triggered()
{
    QString filename = QFileDialog::getOpenFileName( this, tr( "Open Image"), lastPath, tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));

    lastPath = QFileInfo( filename ).absolutePath();

    Mat image;

    if( filename.length() == 0)
    {
        return;
    }
    else
    {
        image = imread( filename.toLocal8Bit().data() );
    }


    if( !image.empty() )
    {
        ui->ImView->setPaintImage(image);
        statusBarMessageLabel->setText( filename + QString( " loaded!"));
    }

}

void MainWindow::on_action_Save_triggered()
{
    QString filename = QFileDialog::getSaveFileName( this, tr("Save Image"),
                                                     lastPath, tr( "Image Files: (*.jpg *.png *.jpeg *.bmp" ) );
    lastPath = QFileInfo( filename ).absolutePath();

    if( filename.length() == 0 ){
        return;
    }

    if( !ui->ImView->isEmpty() ){
        imwrite( filename.toLocal8Bit().data(), ui->ImView->getCurrentImage() );
    }

}

void MainWindow::on_action_Remove_image_triggered()
{

}

void MainWindow::on_actionSVM_triggered()
{
    //Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros( height, width, CV_8UC3 );

    //set up traing data
    float labels[4] = { 1.0, -1.0, -1.0, -1.0 };
    Mat labelsMat( 4, 1, CV_32FC1, labels );

    float trainingData[4][2] = { {501,10}, {255,10}, {501, 255}, {10,501} };
    Mat trainingDataMat( 4, 2, CV_32FC1, trainingData );

    //Set up SVM's parameters
    CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER, 100, 1e-6 );
    params.C = 0.1;

    //Traing the SVM
    CvSVM SVM;
    SVM.train( trainingDataMat, labelsMat, Mat(), Mat(), params );

    Vec3b green( 0, 255, 0 ), blue( 255, 0, 0 );

    //Show the decision regions given by the SVM
    for( int i = 0; i < image.rows; ++i ){
        for( int j = 0; j < image.cols; ++j ){
            Mat sampleMat = ( Mat_<float>( 1, 2 ) << j, i );
            float response = SVM.predict( sampleMat );

            if( response == 1 )
                image.at<Vec3b>( i, j ) = green;
            else if( response == -1 )
                image.at<Vec3b>( i, j ) = blue;
        }
    }
    //show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, Point( 501, 10 ), 5, Scalar( 0, 0, 0 ), thickness, lineType );
    circle( image, Point(255,10  ), 5, Scalar(255,255,255 ), thickness, lineType );
    circle( image, Point( 501, 255 ), 5, Scalar( 255,255,255), thickness, lineType );
    circle( image, Point(10, 501  ), 5, Scalar( 255,255,255), thickness, lineType );

    //show support vectors
    thickness = 2;
    lineType = 8;
    int c = SVM.get_support_vector_count();

    for( int i = 0; i< c; i++ ){
        const float* v = SVM.get_support_vector(i);
        circle( image, Point( (int)v[0], (int)v[1] ), 6, Scalar( 128,128,128), thickness, lineType );
    }

    ui->ImView->setPaintImage( image );


}

void MainWindow::on_actionDFT_triggered()
{
    Mat image = ui->ImView->getCurrentImage();
    if( !image.data ){
        return;
    }
    Mat img;
    cvtColor( image, img, CV_RGB2GRAY );
    Mat padded;
    int m = getOptimalDFTSize( img.rows );
    int n = getOptimalDFTSize( img.cols );

    copyMakeBorder( img, padded, 0, m-img.rows, 0, n-img.cols, BORDER_CONSTANT, Scalar::all(0 ) );

    Mat planes[] = { Mat_<float>( padded ), Mat::zeros( padded.size(), CV_32F ) } ;
    Mat complexl;
    merge( planes, 2, complexl );//add to the expanded another plane with zeros

    dft( complexl, complexl );

    //compute the magnitude and switch to lagarithmic scale
    // log( 1 + sqrt( Re( DFT(I) )^2 + Im( DFT(I) )^2))
    split( complexl, planes );
    magnitude( planes[0], planes[1], planes[0] );
    Mat magl = planes[0];

    magl += Scalar::all(1);
    log( magl, magl );

    //crop the spectrum
    magl = magl( Rect( 0, 0, magl.cols & -2, magl.rows & -2 ) );

    //rearrange the quadrants of Fourier image so that the origin is at the image center
    int cx = magl.cols / 2;
    int cy = magl.rows / 2;

    Mat q0( magl, Rect( 0, 0, cx, cy ) );
    Mat q1( magl, Rect( cx, 0, cx, cy ) );
    Mat q2( magl, Rect( 0, cy, cx, cy ) );
    Mat q3( magl, Rect( cx, cy, cx, cy ) );

    Mat tmp;
    q0.copyTo( tmp );
    q3.copyTo( q0 );
    tmp.copyTo( q3 );

    q1.copyTo( tmp );
    q2.copyTo( q1 );
    tmp.copyTo( q2 );

    //Transform the matrix with float values into
    //a viewable image form( float between values 0 and 1 )
    normalize( magl, magl, 0, 1, CV_MINMAX );
    magl = Mat_<int>( magl * 255 );
    ui->ImView->setPaintImage( magl );

}

void MainWindow::on_actionTest_triggered()
{
    Mat img_1 = ui->ImView->getCurrentImage();
    Mat img_2 = ui->ImView->getSecondImage();

    if( !img_1.data || !img_2.data ){
        return;
    }
    int minHessian = 400;

    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> keypoints_1, keypoints_2;

    detector.detect( img_1, keypoints_1 );
    detector.detect( img_2, keypoints_2 );

    //--Step2:Calculate descriptors( feature vectors )
    SurfDescriptorExtractor extractor;

    Mat descriptors_1, descriptors_2;

    extractor.compute( img_1, keypoints_1,  descriptors_1);
    extractor.compute( img_2, keypoints_2, descriptors_2);

    //--step 3:Matching descriptor vectors with brute force mathcher
    BFMatcher matcher( NORM_L2 );
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    //--Draw mathces
    Mat img_matches;
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );
    ui->ImView->setPaintImage( img_matches );


}

void MainWindow::on_actionObject_detect_triggered()
{
    Mat img_color_1 = ui->ImView->getCurrentImage();
    Mat img_color_2 = ui->ImView->getSecondImage();

    if( !img_color_1.data || !img_color_2.data ){
        return;
    }

    Mat img_1, img_2;
    cvtColor( img_color_1, img_1, CV_RGB2GRAY );
    cvtColor( img_color_2, img_2, CV_RGB2GRAY );

    int minHessian = 400;

    SurfFeatureDetector detector(minHessian);

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    detector.detect( img_1, keypoints_1 );
    detector.detect( img_2, keypoints_2 );

    SurfDescriptorExtractor extractor;

    Mat descriptors_1, descriptors_2;

    extractor.compute( img_1, keypoints_1, descriptors_1 );
    extractor.compute( img_2, keypoints_2, descriptors_2 );

    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    //--Quick calculation of max and min distance between keypoints
    double max_dist = 0; double min_dist = 100;
    for( int i = 0; i < descriptors_1.rows; i++ ){
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;

    }

    printf( "--Max dist: %f \n", max_dist );
    printf( "--Min dist: %f \n", min_dist );

    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_1.rows; i++ ){
        if( matches[i].distance <= 3*min_dist ){
            good_matches.push_back( matches[i]);
        }
    }
    Mat img_matches;
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches,
                Scalar::all(-1), Scalar::all(-1), vector<char>(),
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( int i = 0; i < int(good_matches.size()); i++ ){
        //get the keypoints from the good matches
        obj.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_2[ good_matches[i].trainIdx ].pt);
    }

    Mat H = findHomography( obj, scene, CV_RANSAC );

    //--get the coners from the image_1( the object to be detected )
    std::vector<Point2f> obj_coners(4);
    obj_coners[0] = cvPoint( 0, 0 );
    obj_coners[1] = cvPoint( img_1.cols, 0 );
    obj_coners[2] = cvPoint( img_1.cols, img_1.rows );
    obj_coners[3] = cvPoint( 0, img_1.rows );
    std::vector<Point2f> scene_coners(4);

    perspectiveTransform( obj_coners, scene_coners, H );
    //--Draw lines between the coners
    line( img_matches, scene_coners[0] + Point2f( img_1.cols, 0 ), scene_coners[1]+Point2f(img_1.cols,0 ),
            Scalar(0,255,0), 4);
    line( img_matches, scene_coners[1] + Point2f( img_1.cols, 0 ), scene_coners[2]+Point2f(img_1.cols,0 ),
            Scalar(0,255,0), 4);
    line( img_matches, scene_coners[2] + Point2f( img_1.cols, 0 ), scene_coners[3]+Point2f(img_1.cols,0 ),
            Scalar(0,255,0), 4);
    line( img_matches, scene_coners[3] + Point2f( img_1.cols, 0 ), scene_coners[0]+Point2f(img_1.cols,0 ),
            Scalar(0,255,0), 4);

    ui->ImView->setPaintImage( img_matches );


}

void MainWindow::on_actionConnect_database_triggered()
{
    ConnectDatabaseDialog *dialog = new ConnectDatabaseDialog(this);

    if( dialog->exec() ){
        db = DatabaseManager::createConnection( dialog->ui->databaseTypeBox->currentText(),
                                           dialog->ui->hostNameEdit->text(),
                                           dialog->ui->databaseNameEdit->text(),
                                           dialog->ui->userNameEdit->text(),
                                           dialog->ui->passwordEdit->text() );

    }else{
        return;
    }
    if( !db.isOpen() ){
        statusBarMessageLabel->setText(tr("Open database failed!" ));
    } else {
        statusBarMessageLabel->setText(tr( "Open database successfully!" ));
    }

    delete dialog;
}

void MainWindow::on_actionClose_database_triggered()
{
    if( db.isOpen() ){
        db.close();
    }
    statusBarMessageLabel->setText( tr( "Database closed!"));
}

void MainWindow::on_actionCross_bilateral_triggered()
{
    QElapsedTimer timer;
    timer.start();

    Mat im = ui->ImView->getCurrentImage();
    Mat mask = ui->ImView->getSecondImage();
    if( im.empty() || mask.empty() ){
        statusBarMessageLabel->setText( tr("You must have two image for bilateral filter!"));
        return;
    }

    CrossBilateralFilterDialog *dialog = new CrossBilateralFilterDialog(this);

    if( dialog->exec() ){
       if( im.channels() == 3 ){
            cvtColor( im, im, CV_BGR2GRAY );
        }
        if( mask.channels() == 3 ){
            cvtColor( mask, mask, CV_BGR2GRAY );
        }
        Mat result;
        int wsize = dialog->ui->wsizeBox->value();
        double sigma_space = dialog->ui->sigmaSpaceBox->value();
        double sigma_value = dialog->ui->sigmaValueBox->value();
        Filters filters;
        filters.crossBilateralFilter( im, mask, result, wsize, sigma_space, sigma_value );
        ui->ImView->setPaintImage( result );
        statusBarMessageLabel->setText( tr("cross bilateral filtering elapsed with: ")
                                        + QString::number( timer.elapsed() / 1000.0 ) + " seconds\n");

    }

    delete dialog;
}

void MainWindow::on_actionGuided_Filter_triggered()
{
    QElapsedTimer timer;
    timer.start();

    Mat im = ui->ImView->getCurrentImage();
    Mat mask = ui->ImView->getSecondImage();
    if( im.empty() || mask.empty() ){
        statusBarMessageLabel->setText( tr("You must have two image for bilateral filter!"));
        return;
    }

    GuidedFilterDialog *dialog = new GuidedFilterDialog(this);
    if( dialog->exec() ){
        if( im.channels() == 3 ){
            cvtColor( im, im, CV_BGR2GRAY );
        }
        if( mask.channels() == 3 ){
            cvtColor( mask, mask, CV_BGR2GRAY );
        }
        Mat result;
        int wsize = dialog->ui->wsizeBox->value();
        double regularizationTerm = dialog->ui->regularizationSpinBox->value();
        Filters filters;
        filters.guidedFilter( im, mask, result, wsize, regularizationTerm );
        ui->ImView->setPaintImage( result );

        statusBarMessageLabel->setText( tr("guided filtering elapsed with: ") +
                QString::number( timer.elapsed() / 1000.0 ) + " seconds\n" );
    }


}

void MainWindow::on_actionDIBR_triggered()
{

}
