#include "depthmapgenerationwithkmeansdialog.h"
#include "ui_depthmapgenerationwithkmeansdialog.h"
#include "imtools.h"
#include <QElapsedTimer>
#include <QMessageBox>
#include "ui_mainwindow.h"

DepthMapGenerationWithKmeansDialog::DepthMapGenerationWithKmeansDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DepthMapGenerationWithKmeansDialog)
{
    searcher = NULL;
    ui->setupUi(this);
}

DepthMapGenerationWithKmeansDialog::~DepthMapGenerationWithKmeansDialog()
{
    //delete informationOutput;
    delete ui;
}

void DepthMapGenerationWithKmeansDialog::on_kmeansTrainButton_clicked()
{
    QStringList imlist;
    QStringList depthlist;

    QString imPath = "/home/lang/dataset/NYUDataset/images";
    QString depthPath = "/home/lang/dataset/NYUDataset/depthInYAML";

    imtools::getImListAndDepthList( imPath, depthPath, imlist, depthlist );
    if( imlist.empty() || depthlist.empty() ){
        QMessageBox::information( this, "Training finished",
                                  "The kmeans clustering process has finished" );
    }


    int k = ui->kmeansCentroidsBox->value();
    searcher = new kmeansSearcher( imlist, depthlist, k );
    searcher->train();

    ui_mainWindow->statusBar->showMessage("kmeans training finished!!");
}

void DepthMapGenerationWithKmeansDialog::on_generateButton_clicked()
{
    if( ui_mainWindow->ImView->isEmpty() ){
        return;
    }
    //time counting
    QElapsedTimer timer;
    timer.start();
    if( searcher == NULL ){
        QMessageBox::information( this, "not intialized",
                                  "You should train the kmeans searcher first" );
        return;
    }
    Mat img = ui_mainWindow->ImView->getCurrentImage() ;
    int classNum = searcher->classify( img );
    QString info = "this image belongs to " + QString::number(classNum) + " class";
    QString elapes = "elapsed time with " +
            QString::number( timer.elapsed()/1000.0 ) + tr( " s" );

    output_panel->append( info );
    output_panel->append( elapes );
    output_panel->append( tr(" " ) );
    output_panel->show();
    output_panel->raise();
    output_panel->activateWindow();

    ui_mainWindow->ImView->setPaintImage( searcher->k_depthMaps[classNum] );

}

void DepthMapGenerationWithKmeansDialog::setMainWindowUi(Ui::MainWindow *ui )
{
    ui_mainWindow = ui;
}

void DepthMapGenerationWithKmeansDialog::setOutputPanel( InformationPanel *panel )
{
    output_panel = panel;
}
