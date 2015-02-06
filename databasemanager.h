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
};

#endif // DATABASEMANAGER_H
