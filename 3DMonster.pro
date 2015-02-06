#-------------------------------------------------
#
# Project created by QtCreator 2015-02-04T20:09:26
#
#-------------------------------------------------

QT       += core gui

QT 	 += sql
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = 3DMonster
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    databasemanager.cpp \
    depthmapgeneratingalgorithms.cpp \
    dibr.cpp \
    displaywidget.cpp \
    filters.cpp \
    imtools.cpp \
    kmeanssearcher.cpp \
    knnsearcher.cpp \
    lineextractingalgorithms.cpp \
    parameters.cpp \
    patch.cpp \
    pyramid.cpp \
    statistic.cpp \
    visualword.cpp \
    connectdatabasedialog.cpp \
    crossbilateralfilterdialog.cpp \
    guidedfilterdialog.cpp \
    convertdialoghelper.cpp

HEADERS  += mainwindow.h \
    databasemanager.h \
    depthmapgeneratingalgorithms.h \
    dibr.h \
    displaywidget.h \
    filters.h \
    imtools.h \
    kmeanssearcher.h \
    knnsearcher.h \
    lineextractingalgorithms.h \
    parameters.h \
    patch.h \
    pyramid.h \
    statistic.h \
    visualword.h \
    connectdatabasedialog.h \
    crossbilateralfilterdialog.h \
    guidedfilterdialog.h \
    convertdialoghelper.h

FORMS    += mainwindow.ui \
    connectdatabasedialog.ui \
    crossbilateralfilterdialog.ui \
    guidedfilterdialog.ui

INCLUDEPATH += /usr/local/include/opencv
LIBS += \
/usr/local/lib/libopencv_core.so \
/usr/local/lib/libopencv_highgui.so \
/usr/local/lib/libopencv_ml.so \
/usr/local/lib/libopencv_imgproc.so \
/usr/local/lib/libopencv_features2d.so \
/usr/local/lib/libopencv_nonfree.so \
/usr/local/lib/libopencv_flann.so \
/usr/local/lib/libopencv_calib3d.so \
/usr/local/lib/libopencv_ml.so \
/usr/local/lib/libopencv_gpu.so \
/usr/local/lib/libopencv_objdetect.so \

OTHER_FILES += \
    3DMonster.pro.user
