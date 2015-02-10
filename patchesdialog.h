#ifndef PATCHESDIALOG_H
#define PATCHESDIALOG_H

#include <QDialog>
#include "patch.h"
#include "ui_mainwindow.h"
#include "informationpanel.h"
#include <QSqlQuery>

namespace Ui {
class PatchesDialog;
}

class PatchesDialog : public QDialog
{
    Q_OBJECT

public:
    explicit PatchesDialog(QWidget *parent = 0);
    ~PatchesDialog();

    void setMainWindowUi(Ui::MainWindow *ui);
    void setOutputPanel(InformationPanel *panel);
    void setDatabase(QSqlDatabase *database);
private slots:
    void on_patchSampleButton_clicked();

    void on_showPatchesButton_clicked();

    void on_detectOverlappedPatchesButton_clicked();

    void on_detectFlatsButton_clicked();

private:
    Ui::PatchesDialog *ui;
    Ui::MainWindow *ui_mainWindow;
    InformationPanel *output_panel;

private:

    QSqlDatabase *db;

    Mat targetImageForPatchExtraction;

    //patch class is used to sample patched in a image
    Patch *patchExtracter;

    //patches sampled in pyramid
    //all is 2D array, one row for one layer of pyramid
    std::vector< std::vector<Mat> > patches_array;
    std::vector< std::vector<Point> > coordinates_array;
    std::vector< std::vector<PATCH_TYPE> > overlappedPatchSymbols_array;
    std::vector< std::vector<PATCH_TYPE> > flatPatchSymbols_array;

    //patches sampled in single image
    std::vector<Point> coordinates;
    std::vector<Mat> patches;
    std::vector<PATCH_TYPE> overlappedPatchSymbols;
    std::vector<PATCH_TYPE> flatPatchSymbols;
};

#endif // PATCHESDIALOG_H
