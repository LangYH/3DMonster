#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMenu>
#include <QFileDialog>
#include <QFileInfo>
#include "databasemanager.h"
#include "filters.h"
#include <QElapsedTimer>
#include "crossbilateralfilterdialog.h"
#include "ui_crossbilateralfilterdialog.h"
#include "connectdatabasedialog.h"
#include "ui_connectdatabasedialog.h"
#include "guidedfilterdialog.h"
#include "ui_guidedfilterdialog.h"
#include "imageresizedialog.h"
#include "ui_imageresizedialog.h"
#include <QMessageBox>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    informationOutput = new InformationPanel(this);

    lastPath = "/home/lang/dataset/NYUDataset/images";
    DIBRHelper = new ConvertDialogHelper();
    depthMapGenerationWithKmeansDlg = NULL;
    depthMapGenerationWithKNNDlg = NULL;
    visualWordDlg = NULL;
    pyramidDlg = NULL;
    patchesDlg = NULL;

    //bind signals to slots
    connect( ui->actionE_xit, SIGNAL(triggered()), this, SLOT(close()) );
    connect( ui->action_Remove_image, SIGNAL(triggered()), ui->ImView, SLOT(eraseTopImage()) );
    connect( ui->actionAbout_Qt, SIGNAL(triggered()), qApp, SLOT(aboutQt()) );

    //set context menu
    ui->ImView->addAction( ui->action_Remove_image );

    QMenu *DIBRMenu = new QMenu( this );
    DIBRMenu->addAction( ui->actionSet_as_DIBR_image );
    DIBRMenu->addAction( ui->actionSet_as_DIBR_depthmap );
    ui->actionDIBR_operations->setMenu( DIBRMenu );
    ui->ImView->addAction( ui->actionDIBR_operations );

    ui->ImView->setContextMenuPolicy( Qt::ActionsContextMenu );
    //*****************************************************************


    //connect database: imageset
    db = DatabaseManager::createConnection( "QPSQL","localhost","imageset","lang","lang");
    if( !db.isOpen() )
        exit(1);

    QFont statusBarFont("Times", 15, QFont::Bold );
    statusBar()->setFont( statusBarFont );
    statusBar()->showMessage("Open image to start processing" );

    showMaximized();
    informationOutput->move( this->width(), this->height() );
}

MainWindow::~MainWindow()
{
    if( db.isOpen() ){
        db.close();
    }
    delete informationOutput;
    delete DIBRHelper;
    delete ui;
}

void MainWindow::on_action_Open_triggered()
{
    QString filename = QFileDialog::getOpenFileName( this, tr( "Open Image"), lastPath,
                                            tr("Image Files (*.png *.jpg *.jpeg *.bmp *.yaml)"));

    lastPath = QFileInfo( filename ).absolutePath();

    //QString filename = QFileDialog::getOpenFileName( this, tr( "Open Image"), ".", tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));

    Mat image;

    if( filename.length() == 0)
    {
        return;
    }
    else
    {
        if( QFileInfo(filename).suffix() == "yaml" ){
            FileStorage fs;
            if( fs.open( filename.toLocal8Bit().data(), FileStorage::READ ) ){
                fs[ "depth" ] >> image;
            }
            fs.release();
        }else{
            image = imread( filename.toLocal8Bit().data() );
        }
    }

    if( !image.empty() )
    {
        ui->ImView->setPaintImage(image);
        statusBar()->showMessage( filename + QString( " loaded!") );
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

            if( response == 1.0 )
                image.at<Vec3b>( i, j ) = green;
            else if( response == -1.0 )
                image.at<Vec3b>( i, j ) = blue;
        }
    }
    //show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, Point( 501, 10 ), 5, Scalar( 0, 0, 0 ), thickness, lineType );
    circle( image, Point(255,10  ), 5, Scalar(255,255,255 ), thickness, lineType );
    circle( image, Point( 501, 255 ), 5, Scalar( 255,255,255), thickness, lineType );
    circle( image, Point( 10, 501  ), 5, Scalar( 255,255,255), thickness, lineType );

    //test
    Mat sample1 = ( Mat_<float>(1,2) << 510, 10 );
    Mat sample2 = ( Mat_<float>(1,2) << 520, 10 );
    Mat sample3 = ( Mat_<float>(1,2) << 180, 10 );
    Mat sample4 = ( Mat_<float>(1,2) << 190, 10 );
    float score1 = SVM.predict( sample1, true );
    std::cout << "The score of ( 510, 10 ) is : " << score1 << std::endl;
    float score2 = SVM.predict( sample2, true );
    std::cout << "The score of ( 520, 10 ) is : " << score2 << std::endl;
    float score3 = SVM.predict( sample3, true );
    std::cout << "The score of ( 180, 10 ) is : " << score3 << std::endl;
    float score4 = SVM.predict( sample4, true );
    std::cout << "The score of ( 190, 10 ) is : " << score4 << std::endl;

    //show support vectors
    thickness = 2;
    lineType = 8;
    int c = SVM.get_support_vector_count();

    for( int i = 0; i< c; i++ ){
        const float* v = SVM.get_support_vector(i);
        circle( image, Point( (int)v[0], (int)v[1] ), 6, Scalar( 0, 0, 255), thickness, lineType );
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
    ConnectDatabaseDialog connectDatabaseDlg(this);

    if( connectDatabaseDlg.exec() ){
        db = DatabaseManager::createConnection( connectDatabaseDlg.ui->databaseTypeBox->currentText(),
                                           connectDatabaseDlg.ui->hostNameEdit->text(),
                                           connectDatabaseDlg.ui->databaseNameEdit->text(),
                                           connectDatabaseDlg.ui->userNameEdit->text(),
                                           connectDatabaseDlg.ui->passwordEdit->text() );

    }else{
        return;
    }
    if( !db.isOpen() ){
        statusBar()->showMessage( tr("Open database failed!" ) );
    } else {
        statusBar()->showMessage( tr( "Open database successfully!" ) );
    }

}

void MainWindow::on_actionClose_database_triggered()
{
    if( db.isOpen() ){
        db.close();
    }
    statusBar()->showMessage( tr( "Database closed!") );
}

void MainWindow::on_actionCross_bilateral_triggered()
{
    QElapsedTimer timer;
    timer.start();

    Mat im = ui->ImView->getCurrentImage();
    Mat mask = ui->ImView->getSecondImage();
    if( im.empty() || mask.empty() ){
        statusBar()->showMessage( tr("You must have two image for bilateral filter!") );
        return;
    }

    CrossBilateralFilterDialog crossBilateralFilterDlg(this);

    if( crossBilateralFilterDlg.exec() ){
       if( im.channels() == 3 ){
            cvtColor( im, im, CV_BGR2GRAY );
        }
        if( mask.channels() == 3 ){
            cvtColor( mask, mask, CV_BGR2GRAY );
        }
        Mat result;
        int wsize = crossBilateralFilterDlg.ui->wsizeBox->value();
        double sigma_space = crossBilateralFilterDlg.ui->sigmaSpaceBox->value();
        double sigma_value = crossBilateralFilterDlg.ui->sigmaValueBox->value();
        Filters filters;
        filters.crossBilateralFilter( im, mask, result, wsize, sigma_space, sigma_value );
        ui->ImView->setPaintImage( result );
        statusBar()->showMessage( tr("cross bilateral filtering elapsed with: ")
                                        + QString::number( timer.elapsed() / 1000.0 ) + " seconds\n" );

    }

}

void MainWindow::on_actionGuided_Filter_triggered()
{
    QElapsedTimer timer;
    timer.start();

    Mat im = ui->ImView->getCurrentImage();
    Mat mask = ui->ImView->getSecondImage();
    if( im.empty() || mask.empty() ){
        statusBar()->showMessage( tr("You must have two image for bilateral filter!") );
        return;
    }

    GuidedFilterDialog guidedFilterDlg(this);

    if( guidedFilterDlg.exec() ){
        if( im.channels() == 3 ){
            cvtColor( im, im, CV_BGR2GRAY );
        }
        if( mask.channels() == 3 ){
            cvtColor( mask, mask, CV_BGR2GRAY );
        }
        Mat result;
        int wsize = guidedFilterDlg.ui->wsizeBox->value();
        double regularizationTerm = guidedFilterDlg.ui->regularizationSpinBox->value();
        Filters filters;
        filters.guidedFilter( im, mask, result, wsize, regularizationTerm );
        ui->ImView->setPaintImage( result );

        statusBar()->showMessage( tr("guided filtering elapsed with: ") +
                QString::number( timer.elapsed() / 1000.0 ) + " seconds\n" );
    }

}

void MainWindow::on_actionDIBR_triggered()
{
    DIBRHelper->process();

    ui->ImView->setPaintImage( DIBRHelper->getLastResult() );
}

void MainWindow::on_actionSet_as_DIBR_image_triggered()
{
    if( ui->ImView->isEmpty() )
        return;

    DIBRHelper->setInputImage( ui->ImView->getCurrentImage() );
    statusBar()->showMessage( tr("DIBR image set!") );
}

void MainWindow::on_actionSet_as_DIBR_depthmap_triggered()
{
    if( ui->ImView->isEmpty() )
        return;

    DIBRHelper->setDepthImage( ui->ImView->getCurrentImage() );
    statusBar()->showMessage( tr("DIBR depthmap set!") );
}

void MainWindow::on_actionDepthMap_generation_kmeans_triggered()
{
    if( !depthMapGenerationWithKmeansDlg ){
        depthMapGenerationWithKmeansDlg = new DepthMapGenerationWithKmeansDialog(this);
        depthMapGenerationWithKmeansDlg->setMainWindowUi( ui );
        depthMapGenerationWithKmeansDlg->setOutputPanel( informationOutput );
    }

    depthMapGenerationWithKmeansDlg->move( ui->ImView->pos() );
    depthMapGenerationWithKmeansDlg->show();
    depthMapGenerationWithKmeansDlg->raise();
    depthMapGenerationWithKmeansDlg->activateWindow();

}

void MainWindow::on_actionAbout_triggered()
{
    QMessageBox::about( this, tr("About 3DMonster" ),
                        tr( "<h2>3DMonster 1.1</h2>"
                            "<p>Copyright &copy;LangYH."
                            "<p>3DMonster is an image processing program,mainly "
                            "used in 2D-3D conversion,based on my graduated student project"));
}

void MainWindow::on_actionDepthMap_generation_kNN_triggered()
{
    if( !depthMapGenerationWithKNNDlg ){
        depthMapGenerationWithKNNDlg = new DepthMapGenerationWithKNNDialog(this);
        depthMapGenerationWithKNNDlg->setMainWindowUi( ui );
        depthMapGenerationWithKNNDlg->setOutputPanel( informationOutput );
    }

    depthMapGenerationWithKNNDlg->show();
    depthMapGenerationWithKNNDlg->raise();
    depthMapGenerationWithKNNDlg->activateWindow();

}

void MainWindow::on_actionDepthMap_generation_relative_height_triggered()
{
    QElapsedTimer timer;
    timer.start();
    DIBRHelper->setInputImage( ui->ImView->getCurrentImage() );
    DIBRHelper->generateDepthMapUsingRelativeHeightCue();
    ui->ImView->setPaintImage( DIBRHelper->getLastDepthMapResult() );
    QString info = tr( "relative height cue algorithm elapsed with " ) + QString::number( timer.elapsed() / 1000.0 )
            + tr( " s" );
    statusBar()->showMessage( info );

}

void MainWindow::on_actionResize_image_triggered()
{
    if( ui->ImView->isEmpty() ){
        return;
    }
    Mat original_image = ui->ImView->getCurrentImage();
    ImageResizeDialog dialog(this);
    dialog.ui->widthBox->setValue( original_image.cols );
    dialog.ui->heightBox->setValue( original_image.rows );
    if( dialog.exec() ){
        Mat image;
        cv::resize( original_image, image,
                    cv::Size(dialog.ui->widthBox->value(), dialog.ui->heightBox->value() ) );
        ui->ImView->setPaintImage( image );
    }

}

void MainWindow::on_actionPyramid_triggered()
{
    if( !pyramidDlg ){
        pyramidDlg = new PyramidDialog(this);
        pyramidDlg->setMainWindowUi( ui );
        pyramidDlg->setOutputPanel( informationOutput );
    }

    pyramidDlg->show();
    pyramidDlg->raise();
    pyramidDlg->activateWindow();
}

void MainWindow::on_actionPatches_triggered()
{
    if( !patchesDlg ){
        patchesDlg = new PatchesDialog(this);
        patchesDlg->setMainWindowUi(ui);
        patchesDlg->setOutputPanel( informationOutput );
        patchesDlg->setDatabase( &db );
    }

    patchesDlg->show();
    patchesDlg->raise();
    patchesDlg->activateWindow();

}

void MainWindow::on_actionVisual_Word_Training_triggered()
{
    if( !visualWordDlg ){
        visualWordDlg = new VisualWordDialog(this);
        visualWordDlg->setMainWindowUi( ui );
        visualWordDlg->setDatabase( &db );
    }

    visualWordDlg->show();
    visualWordDlg->raise();
    visualWordDlg->activateWindow();


}
