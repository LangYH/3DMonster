#ifndef VISUALWORDTESTDIALOG_H
#define VISUALWORDTESTDIALOG_H

#include <QDialog>
#include <opencv2/core/core.hpp>
#include <QSqlDatabase>

using namespace cv;

namespace Ui {
class VisualWordTestDialog;
class MainWindow;
}

class VisualWordTestDialog : public QDialog
{
    Q_OBJECT

public:
    explicit VisualWordTestDialog(QWidget *parent = 0);
    ~VisualWordTestDialog();

private:
    Ui::VisualWordTestDialog *ui;
    Ui::MainWindow *ui_mainWindow;
    QSqlDatabase *db;

public:
    void setDatabase(QSqlDatabase *database);
    void setMainWindowUi(Ui::MainWindow *ui);
    void classifyAllPatches(const std::vector<std::vector<Mat> > patches_array, std::vector<std::vector<int> > &result_class, std::vector<std::vector<double> > &svm_score);

private slots:
    void on_visualWordTrainingButton_clicked();
    void on_testFuncButton_clicked();
    void on_visualWordDetectorTrainingButton_clicked();
};

#endif // VISUALWORDTESTDIALOG_H
