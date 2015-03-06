#ifndef VISUALWORDDIALOG_H
#define VISUALWORDDIALOG_H

#include <QDialog>
#include "ui_mainwindow.h"
#include <QSqlDatabase>
#include "visualword.h"

namespace Ui {
class VisualWordDialog;
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

private:
    Ui::VisualWordDialog *ui;
    Ui::MainWindow *ui_mainWindow;

    QSqlDatabase *db;

    VisualWord *vw;

    QStringList D1, D2, N1, N2;

public:
    void setDatabase(QSqlDatabase *database);
    void setMainWindowUi(Ui::MainWindow *ui);
};

#endif // VISUALWORDDIALOG_H
