#ifndef DATABASEMANAGER_H
#define DATABASEMANAGER_H
#include <QSqlDatabase>


class DatabaseManager
{
private:
    QString databaseType;
    QString hostname;
    QString databaseName;
    QString userName;
    QString password;
    QSqlDatabase db;

public:
    DatabaseManager();
    static QSqlDatabase createConnection(QString databaseType, QString hostname, QString databaseName, QString userName, QString password);
    static void loadVisualWordTrainingData(QStringList &I1, QStringList &I2, QStringList &D1, QStringList &D2,
                                            QStringList &N1, QStringList &N2 );
    static void loadNaturalData(QStringList &natural_list);
    static void loadNYUDepthData(QStringList &nyu_depth_list);
    static void loadNYUData(QStringList &nyu_image_list, QStringList &nyu_depth_list);
};

#endif // DATABASEMANAGER_H
