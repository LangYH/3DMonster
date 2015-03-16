#ifndef PYRAMIDDIALOG_H
#define PYRAMIDDIALOG_H

#include <QDialog>
#include "informationpanel.h"
#include "pyramid.h"

namespace Ui {
class PyramidDialog;
class MainWindow;
}

class PyramidDialog : public QDialog
{
    Q_OBJECT

public:
    explicit PyramidDialog(QWidget *parent = 0);
    ~PyramidDialog();

    void setMainWindowUi(Ui::MainWindow *ui);
    void setOutputPanel(InformationPanel *panel);
private slots:
    void on_computeButton_clicked();

    void on_showPyramidButton_clicked();

private:
    Ui::PyramidDialog *ui;
    Ui::MainWindow *ui_mainWindow;
    InformationPanel *output_panel;

private:
    Pyramid *imPyramid;
    std::vector<Mat> pyrs;

};

#endif // PYRAMIDDIALOG_H
