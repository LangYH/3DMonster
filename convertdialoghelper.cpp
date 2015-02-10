#include "convertdialoghelper.h"
#include "depthmapgeneratingalgorithms.h"
#include <QMessageBox>

ConvertDialogHelper::ConvertDialogHelper()
{
    //setting up the application
    DIBRAlgorithm = new DIBR();
}

ConvertDialogHelper::~ConvertDialogHelper()
{
    delete DIBRAlgorithm;
}

bool ConvertDialogHelper::setInputImage( Mat input )
{
    if( !input.empty() )
    {
        input.copyTo( image );
        return true;
    }
    else
    {
        return false;
    }
}

bool ConvertDialogHelper::setDepthImage( Mat inputDepth )
{

    if( !inputDepth.empty() )
    {
        inputDepth.copyTo( depthImage );
        DIBRAlgorithm->depthMapNormalized(depthImage);
        return true;
    }
    else
    {
        return false;
    }
}

void ConvertDialogHelper::generateDepthMapUsingRelativeHeightCue()
{
    if( !image.empty())
    {
        depthImage = DepthMapGeneratingAlgorithms::usingRelativeHeightDepthCue( image );
        depthImage.copyTo( depthImageToDisplay );
        DIBRAlgorithm->depthMapNormalized(depthImage);
    }
}

void ConvertDialogHelper::process()
{
    if( image.empty() || depthImage.empty() )
    {
        QMessageBox::information( NULL, "Empty image or depthmap",
                                  "You haven't set image or depth information" );
        return;
    }
    //imshow( "image", image );waitKey(0);
    //imshow( "depth", depthImage );waitKey(0);

    if( image.size() != depthImage.size())
    {
        resize( image, image, depthImage.size() );
    }
    result = DIBRAlgorithm->execute(image, depthImage );
}

const Mat ConvertDialogHelper::getLastResult() const
{
    return result;
}

const Mat ConvertDialogHelper::getLastDepthMapResult()
{
    return depthImageToDisplay; // just used to display
}
