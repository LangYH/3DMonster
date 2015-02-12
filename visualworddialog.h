#ifndef VISUALWORDDIALOG_H
#define VISUALWORDDIALOG_H

#include <QDialog>

namespace Ui {
class VisualWordDialog;
}

class VisualWordDialog : public QDialog
{
    Q_OBJECT

public:
    explicit VisualWordDialog(QWidget *parent = 0);
    ~VisualWordDialog();

private:
    Ui::VisualWordDialog *ui;
};

#endif // VISUALWORDDIALOG_H
