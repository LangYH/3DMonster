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
    convertdialoghelper.cpp \
    depthmapgenerationwithkmeansdialog.cpp \
    informationpanel.cpp \
    depthmapgenerationwithknndialog.cpp \
    imageresizedialog.cpp \
    pyramiddialog.cpp \
    patchesdialog.cpp \
    visualworddialog.cpp \
    visualworddictionary.cpp \
    depthmakerwithvisualword.cpp \
    visualword2.cpp \
    visualwordtestdialog.cpp \
    visualworddictionary2.cpp

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
    convertdialoghelper.h \
    depthmapgenerationwithkmeansdialog.h \
    informationpanel.h \
    depthmapgenerationwithknndialog.h \
    imageresizedialog.h \
    pyramiddialog.h \
    patchesdialog.h \
    visualworddialog.h \
    visualworddictionary.h \
    depthmakerwithvisualword.h \
    visualword2.h \
    CustomType.h \
    visualwordtestdialog.h \
    visualworddictionary2.h

FORMS    += mainwindow.ui \
    connectdatabasedialog.ui \
    crossbilateralfilterdialog.ui \
    guidedfilterdialog.ui \
    depthmapgenerationwithkmeansdialog.ui \
    informationpanel.ui \
    depthmapgenerationwithknndialog.ui \
    imageresizedialog.ui \
    pyramiddialog.ui \
    patchesdialog.ui \
    visualworddialog.ui \
    visualwordtestdialog.ui

INCLUDEPATH += /usr/local/include/opencv
LIBS += \
/usr/local/lib/libopencv_core.so \
/usr/local/lib/libopencv_highgui.so \
/usr/local/lib/libopencv_ml.so \
/usr/local/lib/libopencv_imgproc.so \
/usr/local/lib/libopencv_features2d.so \
/usr/local/lib/libopencv_flann.so \
/usr/local/lib/libopencv_calib3d.so \
/usr/local/lib/libopencv_ml.so \
/usr/local/lib/libopencv_gpu.so \
/usr/local/lib/libopencv_objdetect.so \

QMAKE_CXXFLAGS += -fopenmp
LIBS += -fopenmp

OTHER_FILES += \
    3DMonster.pro.user
