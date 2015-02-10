#ifndef INFORMATIONPANEL_H
#define INFORMATIONPANEL_H

#include <QDialog>

namespace Ui {
class InformationPanel;
}

class InformationPanel : public QDialog
{
    Q_OBJECT

public:
    explicit InformationPanel(QWidget *parent = 0);
    ~InformationPanel();

    void clean();
    void append(const QString &info);

private:
    Ui::InformationPanel *ui;
};

#endif // INFORMATIONPANEL_H
