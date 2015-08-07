#ifndef DEPTHMAPGENERATIONWIHTKNNDIALOG_H
#define DEPTHMAPGENERATIONWIHTKNNDIALOG_H

#include <QDialog>
#include "informationpanel.h"

namespace Ui {
class DepthMapGenerationWithKNNDialog;
class MainWindow;
}

class DepthMapGenerationWithKNNDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DepthMapGenerationWithKNNDialog(QWidget *parent = 0);
    ~DepthMapGenerationWithKNNDialog();

    void setMainWindowUi(Ui::MainWindow *ui);

    void setOutputPanel(InformationPanel *panel);

private slots:
    void on_generateButton_clicked();

    void on_actionDepthMap_generation_kNN_triggered();

private:
    Ui::DepthMapGenerationWithKNNDialog *ui;
    Ui::MainWindow *ui_mainWindow;

    //information panel:using to print app information, pointer comes from mainwindow
    InformationPanel *output_panel;

};

#endif // DEPTHMAPGENERATIONWIHTKNNDIALOG_H
