#ifndef GUIDEDFILTERDIALOG_H
#define GUIDEDFILTERDIALOG_H

#include <QDialog>

namespace Ui {
class GuidedFilterDialog;
}

class GuidedFilterDialog : public QDialog
{
    Q_OBJECT

public:
    explicit GuidedFilterDialog(QWidget *parent = 0);
    ~GuidedFilterDialog();

public:
    Ui::GuidedFilterDialog *ui;
};

#endif // GUIDEDFILTERDIALOG_H
