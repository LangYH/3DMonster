#ifndef IMAGERESIZEDIALOG_H
#define IMAGERESIZEDIALOG_H

#include <QDialog>

namespace Ui {
class ImageResizeDialog;
}

class ImageResizeDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ImageResizeDialog(QWidget *parent = 0);
    ~ImageResizeDialog();

public:
    Ui::ImageResizeDialog *ui;
};

#endif // IMAGERESIZEDIALOG_H
