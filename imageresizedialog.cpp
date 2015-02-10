#include "imageresizedialog.h"
#include "ui_imageresizedialog.h"

ImageResizeDialog::ImageResizeDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ImageResizeDialog)
{
    ui->setupUi(this);
}

ImageResizeDialog::~ImageResizeDialog()
{
    delete ui;
}
