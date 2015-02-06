#include "crossbilateralfilterdialog.h"
#include "ui_crossbilateralfilterdialog.h"

CrossBilateralFilterDialog::CrossBilateralFilterDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::CrossBilateralFilterDialog)
{
    ui->setupUi(this);
}

CrossBilateralFilterDialog::~CrossBilateralFilterDialog()
{
    delete ui;
}
