#ifndef CONNECTDATABASEDIALOG_H
#define CONNECTDATABASEDIALOG_H

#include <QDialog>

namespace Ui {
class ConnectDatabaseDialog;
}

class ConnectDatabaseDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ConnectDatabaseDialog(QWidget *parent = 0);
    ~ConnectDatabaseDialog();

public:
    Ui::ConnectDatabaseDialog *ui;
};

#endif // CONNECTDATABASEDIALOG_H
