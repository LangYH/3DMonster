#include "patchesdialog.h"
#include "ui_patchesdialog.h"
#include <QMessageBox>

PatchesDialog::PatchesDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PatchesDialog)
{
    ui_mainWindow = NULL;
    patchExtracter = NULL;
    ui->setupUi(this);
}

PatchesDialog::~PatchesDialog()
{
    delete ui;
}

void PatchesDialog::setMainWindowUi(Ui::MainWindow *ui )
{
   ui_mainWindow = ui;
}

void PatchesDialog::setOutputPanel( InformationPanel *panel )
{
    output_panel = panel;
}

void PatchesDialog::setDatabase( QSqlDatabase *database )
{
    db = database;
    if( patchExtracter != NULL ){
        delete patchExtracter;
    }
}

void PatchesDialog::on_patchSampleButton_clicked()
{
    if( ui_mainWindow->ImView->isEmpty() ){
        return;
    }

    targetImageForPatchExtraction = ui_mainWindow->ImView->getCurrentImage();

    int init_patch_size = ui->patchSizeBox->value();
    int samples_number = ui->sampleNumberBox->value();

    if( patchExtracter != NULL ){
        delete patchExtracter;
    }

    patchExtracter = new Patch( init_patch_size );

    patchExtracter->randomSamplePatches( targetImageForPatchExtraction,
                                         patches, coordinates, samples_number );

    // intialize overlapped symbols and flat symbols
    std::vector<PATCH_TYPE>().swap( overlappedPatchSymbols );
    overlappedPatchSymbols.assign( samples_number, POSITIVE );
    std::vector<PATCH_TYPE>().swap( flatPatchSymbols );
    flatPatchSymbols.assign( samples_number, POSITIVE );


    //ui operation
    ui->detectOverlappedPatchesButton->setEnabled(true);
    ui->positivePatchesCheckBox->setEnabled(true);

    ui->overlappedPatchCheckBox->setEnabled(false);
    ui->overlappedPatchCheckBox->setChecked(false);
    ui->flatPatchCheckBox->setEnabled(false);
    ui->flatPatchCheckBox->setChecked(false);
    //****************************************************

    ui_mainWindow->statusBar->showMessage( tr( "Sampling process finished!\n" ) );

}

void PatchesDialog::on_showPatchesButton_clicked()
{
    int patchesToShow = ui->samplesToShowBox->value();
    int patchSize = ui->patchSizeBox->value();

    if( patches.size() < static_cast<unsigned int>(patchesToShow) ){
        QMessageBox::warning( this, tr( "Not enough patches" ),
                              QString( tr("There only " ) + QString::number(patches.size() )
                                       +tr(" patches" ) ) );
        return;
    }

    Mat image = targetImageForPatchExtraction.clone();
    if( image.channels() == 1 ){
        cvtColor(image, image, CV_GRAY2BGR);
    }

    Patch::drawFrameForPatchesInImage( image, coordinates,
                                       overlappedPatchSymbols, flatPatchSymbols,
                                       ui->positivePatchesCheckBox->isChecked(),
                                       ui->overlappedPatchCheckBox->isChecked(),
                                       ui->flatPatchCheckBox->isChecked(),
                                       patchesToShow, patchSize );

    int nbr_positive_patches = 0;
    for( unsigned int i = 0; i < patches.size(); i++ ){
        if( overlappedPatchSymbols[i] == POSITIVE && flatPatchSymbols[i] == POSITIVE )
            nbr_positive_patches += 1;
    }
    ui_mainWindow->ImView->setPaintImage( image );
    ui_mainWindow->statusBar->showMessage( tr( "number of positive patches: " ) +
                             QString::number( nbr_positive_patches ) );

}

void PatchesDialog::on_detectOverlappedPatchesButton_clicked()
{
    if( patches.empty() ){
        ui_mainWindow->statusBar->showMessage(
                    tr( "You don't have any samples yet.Sample first,please! "));
        return;
    }
    patchExtracter->detectOverlappedPatches( patches,overlappedPatchSymbols,
                                             ui->overlappedThresholdBox->value(),
                                             CROSS_CORRELATION );

    //count how many pathes have been detected as overlappes
    int nbr_overlap_patches= 0;
    for( std::vector<PATCH_TYPE>::iterator iter = overlappedPatchSymbols.begin();
         iter < overlappedPatchSymbols.end(); iter++ ){
        if( *iter == OVERLAP )
             nbr_overlap_patches += 1;
    }

    ui->overlappedPatchCheckBox->setEnabled(true);

    ui_mainWindow->statusBar->showMessage( tr( "Overlapped patches detection completed: " )
                                           +  QString::number( nbr_overlap_patches) +
                             " overlapped patches has been detected!" );

}

void PatchesDialog::on_detectFlatsButton_clicked()
{

    if( patches.empty() ){
        ui_mainWindow->statusBar->showMessage(
                    tr( "You don't have any samples yet.Sample first,please! "));
        return;
    }

    patchExtracter->detectFlatPatches( patches, flatPatchSymbols, ui->flatsThresholdSpinBox->value(),
                                       DEVIATION);

    //count how many pathes have been detected as flats
    int nbr_flat_patches = 0;
    for( std::vector<PATCH_TYPE>::iterator iter = flatPatchSymbols.begin();
         iter < flatPatchSymbols.end(); iter++ ){
        if( *iter == FLAT )
            nbr_flat_patches += 1;
    }

    ui->flatPatchCheckBox->setEnabled(true);
    ui_mainWindow->statusBar->showMessage( tr( "Flat patches detection completed: " )
                                           +  QString::number( nbr_flat_patches ) +
                             " flat patches has been detected!" );

}
