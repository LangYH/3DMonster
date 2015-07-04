#include "databasemanager.h"
#include <QStringList>
#include <QDir>
#include <QSqlQuery>
#include <QVariant>
#include <assert.h>

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


void DatabaseManager::loadVisualWordTrainingData(QStringList &I1, QStringList &I2,
                                                 QStringList &D1, QStringList &D2,
                                                 QStringList &N1, QStringList &N2)
{
    QStringList nyu_image_list;
    QStringList nyu_depth_list;
    QStringList natural_list;

    loadNYUData( nyu_image_list, nyu_depth_list );
    loadNaturalData( natural_list );

    for( int i = 0; i < nyu_image_list.size() / 2; i++ ){
        I1.push_back( nyu_image_list[i] );
    }

    for( int i = nyu_image_list.size() / 2; i < nyu_image_list.size(); i++ ){
        I2.push_back( nyu_image_list[i] );
    }

    for( int i = 0; i < nyu_depth_list.size() / 2; i++ ){
        D1.push_back( nyu_depth_list[i] );
    }

    for( int i = nyu_depth_list.size() / 2; i < nyu_depth_list.size(); i++ ){
        D2.push_back( nyu_depth_list[i] );
    }

    for( int i = 0; i < natural_list.size() / 2; i++ ){
        N1.push_back( natural_list[i] );
    }

    for( int i = natural_list.size() / 2; i < natural_list.size(); i++ ){
        N2.push_back( natural_list[i] );
    }

}

void DatabaseManager::loadNYUData( QStringList &nyu_image_list,
                                   QStringList &nyu_depth_list )
{
    QSqlQuery query;
    QString SQL_query_command = "SELECT image_path, image_name, depth_path, depth_name "
                        " FROM nyu "
            " ORDER BY id ;";
    query.prepare( SQL_query_command );
    assert( query.exec());

    QString image_path, depth_path;
    while( query.next() ){
        image_path = query.value("image_path").toString() +
                QDir::separator() + query.value("image_name").toString();
        depth_path = query.value("depth_path" ).toString() +
                QDir::separator() + query.value("depth_name").toString();
        nyu_image_list.push_back( image_path );
        nyu_depth_list.push_back( depth_path );
    }
}

void DatabaseManager::loadNYUDepthData( QStringList &nyu_depth_list )
{
    QSqlQuery query;
    QString SQL_query_command = "SELECT depth_path, depth_name "
                        " FROM nyu ;";
    query.prepare( SQL_query_command );
    assert( query.exec());
    QString path, name, full_path;
    while( query.next() ){
        path = query.value("depth_path" ).toString();
        name = query.value("depth_name").toString();
        full_path = path + QDir::separator() + name;
        nyu_depth_list.push_back( full_path );
    }

}

void DatabaseManager::loadNaturalData( QStringList &natural_list )
{
    //get all natural images
    QSqlQuery query;
    QString command = "SELECT image_path, image_name "
            " FROM natural_image ;";
    //        " WHERE id % 6 = 0 "
    //        " ORDER BY id "
    //        " LIMIT 1000; ";

    assert( query.exec(command) );

    while( query.next() ){
        QString path = query.value(0).toString();
        QString name = query.value(1).toString();
        QString full_path = path + QDir::separator() + name;
        natural_list.push_back( full_path );
    }
}


