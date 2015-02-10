#include "pyramiddialog.h"
#include "ui_pyramiddialog.h"
#include <QMessageBox>

PyramidDialog::PyramidDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PyramidDialog)
{
    imPyramid = NULL;
    ui->setupUi(this);
}

PyramidDialog::~PyramidDialog()
{
    delete ui;
}

void PyramidDialog::setMainWindowUi(Ui::MainWindow *ui )
{
   ui_mainWindow = ui;
}

void PyramidDialog::setOutputPanel( InformationPanel *panel )
{
    output_panel = panel;
}

void PyramidDialog::on_computeButton_clicked()
{
    if( ui_mainWindow->ImView->isEmpty() ){
        return;
    }
    int init_octaves = ui->octavesBox->value();
    int init_octaveLayers = ui->octaveLayersBox->value();
    double init_sigma = ui->sigmaBox->value();

    if( imPyramid != NULL )
        delete imPyramid;
    imPyramid = new Pyramid( init_octaves, init_octaveLayers, init_sigma );

    Mat originalImage = ui_mainWindow->ImView->getCurrentImage();

    imPyramid->buildGaussianPyramid( originalImage, pyrs );

    ui_mainWindow->statusBar->showMessage( "Image pyramid builded!");

    return;


}

void PyramidDialog::on_showPyramidButton_clicked()
{
    unsigned int numOfImageToShow = ui->showPyramidLayersBox->value();

    if( pyrs.size() < numOfImageToShow ){
        QMessageBox::warning( this, tr("Not enough Layers"),
                              QString(tr( "The pyramid has only " )+QString::number(pyrs.size())
                                      + " layers" ) );
        return;
    }else{
        for( unsigned int i = 0; i < numOfImageToShow; i++ ){
            ui_mainWindow->ImView->setPaintImage( pyrs[i]);
        }

        return;
    }

}
