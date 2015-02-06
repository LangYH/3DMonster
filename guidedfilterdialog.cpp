#include "guidedfilterdialog.h"
#include "ui_guidedfilterdialog.h"

GuidedFilterDialog::GuidedFilterDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GuidedFilterDialog)
{
    ui->setupUi(this);
}

GuidedFilterDialog::~GuidedFilterDialog()
{
    delete ui;
}
