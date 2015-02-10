#ifndef DEPTHMAPGENERATIONWITHKMEANSDIALOG_H
#define DEPTHMAPGENERATIONWITHKMEANSDIALOG_H

#include <QDialog>
#include "kmeanssearcher.h"
#include "displaywidget.h"
#include "ui_mainwindow.h"
#include "informationpanel.h"

namespace Ui {
class DepthMapGenerationWithKmeansDialog;
}

class DepthMapGenerationWithKmeansDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DepthMapGenerationWithKmeansDialog(QWidget *parent = 0);
    ~DepthMapGenerationWithKmeansDialog();

private slots:
    void on_kmeansTrainButton_clicked();

    void on_generateButton_clicked();

private:
    Ui::DepthMapGenerationWithKmeansDialog *ui;

private:
    kmeansSearcher *searcher;

    //the ui pointer indicate the mainwindow's ui
    Ui::MainWindow *ui_mainWindow;

    //information panel
    InformationPanel *output_panel;

public:

    void setMainWindowUi(Ui::MainWindow *ui);
    void setOutputPanel(InformationPanel *panel);

};

#endif // DEPTHMAPGENERATIONWITHKMEANSDIALOG_H
