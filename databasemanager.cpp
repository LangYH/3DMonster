#include "databasemanager.h"

DatabaseManager::DatabaseManager()
{
}

QSqlDatabase DatabaseManager::createConnection(QString databaseType,
                                               QString hostname,
                                               QString databaseName,
                                               QString userName,
                                               QString password )
{
    QSqlDatabase db = QSqlDatabase::addDatabase( databaseType );
    db.setHostName( hostname );
    db.setDatabaseName( databaseName );
    db.setUserName( userName );
    db.setPassword( password );

    db.open();

    return db;

}
