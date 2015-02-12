#include "visualworddialog.h"
#include "ui_visualworddialog.h"

VisualWordDialog::VisualWordDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::VisualWordDialog)
{
    ui->setupUi(this);
}

VisualWordDialog::~VisualWordDialog()
{
    delete ui;
}
