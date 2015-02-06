#include "connectdatabasedialog.h"
#include "ui_connectdatabasedialog.h"

ConnectDatabaseDialog::ConnectDatabaseDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ConnectDatabaseDialog)
{
    ui->setupUi(this);
}

ConnectDatabaseDialog::~ConnectDatabaseDialog()
{
    delete ui;
}
