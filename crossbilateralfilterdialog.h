#ifndef CROSSBILATERALFILTERDIALOG_H
#define CROSSBILATERALFILTERDIALOG_H

#include <QDialog>

namespace Ui {
class CrossBilateralFilterDialog;
}

class CrossBilateralFilterDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CrossBilateralFilterDialog(QWidget *parent = 0);
    ~CrossBilateralFilterDialog();

public:
    Ui::CrossBilateralFilterDialog *ui;
};

#endif // CROSSBILATERALFILTERDIALOG_H
