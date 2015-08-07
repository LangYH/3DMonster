#include "filters.h"
#include "imtools.h"

Filters::Filters()
{
}

void Filters::crossBilateralFilter(Mat const &srcImage, const Mat &srcmask,
                          Mat &dstImage, int wsize, double sigma_space, double sigma_value )
{
    int cn = srcImage.channels();
    int i, j, k, maxk, radius;
    Size size = srcImage.size();

    if( srcImage.empty() || srcmask.empty() )
        return;

    Mat mask;
    if( srcmask.size() != size ){
        resize( srcmask, mask, size );
    }else{
        srcmask.copyTo( mask );
    }

    //convert mask to srcImage's type
    mask.convertTo( mask, srcImage.type() );

    if( dstImage.empty() )
        dstImage.create( size, srcImage.type() );


    if( sigma_value <= 0 )
        sigma_value = 1;
    if( sigma_space <= 0 )
        sigma_space = 1;

    double gauss_value_coeff = -0.5 / ( sigma_value * sigma_value );
    double gauss_space_coeff = -0.5 / ( sigma_space * sigma_space );

    if( wsize <= 0 )
        radius = cvRound( sigma_space * 1.5 );
    else
        radius = wsize/2;
    radius = MAX( radius, 1 );
    wsize = radius * 2 + 1;

    Mat temp;
    copyMakeBorder( srcImage, temp, radius, radius, radius, radius, BORDER_DEFAULT );
    copyMakeBorder( mask, mask, radius, radius, radius, radius, BORDER_DEFAULT );

    std::vector<float> _value_weight( cn * 256 );
    std::vector<float> _space_weight( wsize*wsize );
    std::vector<int> _space_ofs( wsize*wsize );
    float* value_weight = &_value_weight[0];
    float* space_weight = &_space_weight[0];
    int* space_ofs = &_space_ofs[0];

    for( i = 0; i < 256 * cn; i++ )
        value_weight[i] = (float)std::exp( i*i*gauss_value_coeff );

    for( i = -radius, maxk=0; i <= radius; i++ ){
        for( j = -radius; j <= radius; j++ ){
            double r = std::sqrt( (double)i*i + (double)j*j );
            if( r > radius )
                continue;
            space_weight[maxk] = (float)std::exp( r*r*gauss_space_coeff );
            space_ofs[maxk++] = (int)(i*temp.step + j * cn );
        }
    }

    for( i = 0; i < size.height; i++ ){
        const uchar* sptr = temp.data + ( i+radius)*temp.step + radius*cn;
        const uchar* mptr = mask.data + ( i+radius)*mask.step + radius*cn;
        uchar* dptr = dstImage.data + i*dstImage.step;

        if( cn == 1 ){
            for( j = 0; j < size.width; j++ ){
                float sum = 0, wsum = 0;
                int val0 = mptr[j];
                for( k = 0; k < maxk; k++ ){
                    int val = mptr[ j + space_ofs[k]];
                    float w = space_weight[k]*value_weight[std::abs( val - val0 ) ];
                    sum += sptr[j+space_ofs[k]] * w;
                    wsum += w;
                }
                dptr[j] = (uchar)cvRound( sum/wsum );
            }
        }
        else{
            assert( cn==3 );
            for( j = 0; j< size.width * 3; j += 3 ){
                float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
                int b0 = mptr[j], g0 = mptr[j+1], r0 = mptr[j+2];
                for( k = 0; k < maxk; k++ ){
                    const uchar* sptr_k = sptr + j + space_ofs[k];
                    const uchar* mptr_k = mptr + j + space_ofs[k];
                    int b = mptr_k[0], g = mptr_k[1], r = mptr_k[2];
                    float w = space_weight[k] * value_weight[ std::abs(b-b0) +
                            std::abs(g-g0)+std::abs(r-r0 ) ];
                    sum_b += sptr_k[0]*w; sum_g += sptr_k[1]*w; sum_r += sptr_k[2]*w;
                    wsum += w;
                }
                wsum = 1.f/wsum;
                b0 = cvRound( sum_b * wsum );
                g0 = cvRound( sum_g * wsum );
                r0 = cvRound( sum_r * wsum );
                dptr[j] = (uchar)b0;
                dptr[j+1] = ( uchar )g0;
                dptr[j+2] = (uchar)r0;
            }
        }
    }
}

void Filters::guidedFilter( Mat const &srcImage, Mat const &mask, Mat &dstImage,
                   int wsize, double regularzationTerm )
{
    if( srcImage.empty() || mask.empty() ){
        return;
    }

    Mat P = srcImage.clone();
    Mat I = mask.clone();

    if( I.size() != P.size() ){
        resize( I, I, P.size() );
    }
    if( P.channels() != 1 ){
        cvtColor( P, P, CV_BGR2GRAY );
    }
    if( I.channels() != 1 ){
        cvtColor( I, I, CV_BGR2GRAY );
    }
    P.convertTo( P, CV_64FC1);
    I.convertTo( I, CV_64FC1);

    int radius;
    if( wsize >0 ){
        radius = wsize / 2;
        wsize = radius * 2 + 1;
    }else{
        radius = 1;
        wsize = 3;
    }

    if( regularzationTerm <= 0 ){
        regularzationTerm = 0.1 * 0.1;
    }
    //Step one: compute correspondense matrix of I and p
    Mat meanI, meanP;
    blur( I, meanI,Size( wsize, wsize ) );
    blur( P, meanP, Size( wsize, wsize ) );
    Mat corrI, corrIP;
    multiply( I, I, corrI );
    multiply( I, P, corrIP);
    blur( corrI, corrI, Size( wsize, wsize ) );
    blur( corrIP, corrIP, Size( wsize, wsize ) );
    //Step two: compute selfvariance matrix varI and covariance matrix cov covIP
    Mat varI, covIP;
    varI = corrI - meanI.mul( meanI );
    covIP = corrIP - meanI.mul( meanP );
    //Step three: compute a and b martix
    Mat a, b;
    divide( covIP, ( varI + regularzationTerm ), a );
    b = meanP - a.mul( meanI );
    //Step four: mean blur matrix a and b
    blur( a, a, Size( wsize, wsize ) );
    blur( b, b, Size( wsize, wsize ) );
    //Step five: compute target image using a and b: q = a * I + b
    Mat Imeana;
    multiply( a, I, Imeana );
    Mat result = Imeana + b;
    imtools::matrixNormalize( result, dstImage );

}
