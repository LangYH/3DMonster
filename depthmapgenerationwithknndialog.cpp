#include "depthmapgenerationwithknndialog.h"
#include "ui_depthmapgenerationwithknndialog.h"
#include "depthmapgeneratingalgorithms.h"
#include <QElapsedTimer>

DepthMapGenerationWithKNNDialog::DepthMapGenerationWithKNNDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DepthMapGenerationWithKNNDialog)
{
    ui->setupUi(this);
}

DepthMapGenerationWithKNNDialog::~DepthMapGenerationWithKNNDialog()
{
    delete ui;
}

void DepthMapGenerationWithKNNDialog::setMainWindowUi(Ui::MainWindow *ui )
{
   ui_mainWindow = ui;
}

void DepthMapGenerationWithKNNDialog::setOutputPanel( InformationPanel *panel )
{
    output_panel = panel;
}

void DepthMapGenerationWithKNNDialog::on_generateButton_clicked()
{
    if( ui_mainWindow->ImView->isEmpty() ){
        return;
    }
    QElapsedTimer timer;
    timer.start();

    Mat depthMap;
    int k = ui->kOfkNNBox->value();
    DepthMapGeneratingAlgorithms::usingkNNWithHOG( ui_mainWindow->ImView->getCurrentImage(), depthMap, k );
    ui_mainWindow->ImView->setPaintImage( depthMap );
    QString info = tr( "kNN algorithm elapsed with " ) + QString::number( timer.elapsed() / 1000.0 )
            + tr( " s" );

    output_panel->append( info );
    output_panel->append( tr(" " ) );
    output_panel->show();
    output_panel->raise();
    output_panel->activateWindow();

}

void DepthMapGenerationWithKNNDialog::on_actionDepthMap_generation_kNN_triggered()
{
    return;

}
