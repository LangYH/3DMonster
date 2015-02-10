#include "informationpanel.h"
#include "ui_informationpanel.h"

InformationPanel::InformationPanel(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::InformationPanel)
{
    ui->setupUi(this);
}

InformationPanel::~InformationPanel()
{
    delete ui;
}

void InformationPanel::append( const QString &info )
{
    ui->textBrowser->append( info );
}

void InformationPanel::clean( )
{
    ui->textBrowser->clear();
}
