#include "visualwordtestdialog.h"
#include "ui_visualwordtestdialog.h"
#include "visualword2.h"
#include <QElapsedTimer>

VisualWordTestDialog::VisualWordTestDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::VisualWordTestDialog)
{
    ui->setupUi(this);
    ui_mainWindow = NULL;
    db = NULL;
}

VisualWordTestDialog::~VisualWordTestDialog()
{
    delete ui;
}

void VisualWordTestDialog::setDatabase( QSqlDatabase *database )
{
    db = database;
}

void VisualWordTestDialog::setMainWindowUi(Ui::MainWindow *ui )
{
    ui_mainWindow = ui;
}

void VisualWordTestDialog::on_visualWordTrainingButton_clicked()
{

    QElapsedTimer timer;
    timer.start();
    VisualWord2 *vw2 = new VisualWord2;
    vw2->parallelTrain();
    std::cout << "Using time for multithread: " << timer.elapsed() / 1000.0
              << " seconds " << std::endl;

    //timer.restart();
    //vw2->train();
    //std::cout << "Using time for single thread: " << timer.elapsed() / 1000.0
    //          << " seconds " << std::endl;
    delete vw2;
}

void VisualWordTestDialog::on_testFuncButton_clicked()
{
    VisualWord2 *vw2 = new VisualWord2;

    delete vw2;
}
