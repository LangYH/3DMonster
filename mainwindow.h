#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <QLabel>
#include <QSqlDatabase>
#include "convertdialoghelper.h"
#include "depthmapgenerationwithkmeansdialog.h"
#include "depthmapgenerationwithknndialog.h"
#include "visualworddialog.h"
#include "pyramiddialog.h"
#include "patchesdialog.h"
#include "informationpanel.h"

using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_action_Open_triggered();

    void on_action_Save_triggered();

    void on_action_Remove_image_triggered();

    void on_actionSVM_triggered();

    void on_actionDFT_triggered();

    void on_actionTest_triggered();

    void on_actionObject_detect_triggered();

    void on_actionConnect_database_triggered();

    void on_actionClose_database_triggered();

    void on_actionCross_bilateral_triggered();

    void on_actionGuided_Filter_triggered();

    void on_actionDIBR_triggered();

    void on_actionSet_as_DIBR_image_triggered();

    void on_actionSet_as_DIBR_depthmap_triggered();

    void on_actionDepthMap_generation_kmeans_triggered();

    void on_actionAbout_triggered();

    void on_actionDepthMap_generation_kNN_triggered();

    void on_actionDepthMap_generation_relative_height_triggered();

    void on_actionResize_image_triggered();

    void on_actionPyramid_triggered();

    void on_actionPatches_triggered();

    void on_actionVisual_Word_Training_triggered();

private:
    Ui::MainWindow *ui;
    DepthMapGenerationWithKmeansDialog *depthMapGenerationWithKmeansDlg;
    DepthMapGenerationWithKNNDialog *depthMapGenerationWithKNNDlg;
    VisualWordDialog *visualWordDlg;
    PyramidDialog *pyramidDlg;
    PatchesDialog *patchesDlg;
    InformationPanel *informationOutput;

private:
    //database infomation
    QString databaseType;
    QString hostname;
    QString databaseName;
    QString userName;
    QString password;
    QSqlDatabase db;

private:
    QString lastPath;
    ConvertDialogHelper *DIBRHelper;
};

#endif // MAINWINDOW_H
