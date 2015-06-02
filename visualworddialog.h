#ifndef VISUALWORDDIALOG_H
#define VISUALWORDDIALOG_H

#include <QDialog>
#include <QSqlDatabase>
#include "visualword.h"
#include "visualworddictionary.h"


namespace Ui {
class VisualWordDialog;
class MainWindow;
}

class VisualWordDialog : public QDialog
{
    Q_OBJECT

public:
    explicit VisualWordDialog(QWidget *parent = 0);
    ~VisualWordDialog();

private slots:
    void on_loadDataButton_clicked();

    void on_kmeansClusteringButton_clicked();

    void on_testButton_clicked();

    void on_svmTrainingButton_clicked();

    void on_showOneClassButton_clicked();

    void on_trainAllButton_clicked();

    void on_findSimilarButton_clicked();

    void on_computeDeviationButton_clicked();

    void on_trainMultipleSVMClassifierButton_clicked();

    void on_prepareDataButton_clicked();

    void on_testClassifierButton_clicked();

    void on_convertAImageButton_clicked();

private:
    Ui::VisualWordDialog *ui;
    Ui::MainWindow *ui_mainWindow;

    QSqlDatabase *db;

    VisualWord *vw;
    VisualWordDictionary *dict;

    QStringList D1, D2, N1, N2;

public:
    void setDatabase(QSqlDatabase *database);
    void setMainWindowUi(Ui::MainWindow *ui);
    void classifyAllPatches(const std::vector<std::vector<Mat> > patches_array, std::vector<std::vector<int> > &result_class, std::vector<std::vector<double> > &svm_score);
};

#endif // VISUALWORDDIALOG_H
