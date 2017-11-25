#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <queue>
#include <iostream>
#include <cmath>

#include "face_detection.h"
#include "face_alignment.h"

using namespace cv;
using namespace std;
//#define MAX(a,b) ( ((a)>(b)) ? (a):(b) )
//#define MIN(a,b) ( ((a)>(b)) ? (b):(a) )
string MODEL_DIR = "../../model/";
int detectFace(Mat imgG,seeta::FaceDetection &detector,Rect &face,seeta::FaceAlignment &point_detector,Point marks[]);

int readLists(const char *  fileName,vector<string> & lists,string prefix="");
void filiterEdge(Mat  ed,Mat & fi);

Rect RectChangeBase(const Rect &big,const Rect &small);
Rect RectChangeBase(const Mat &pic,const Rect &small);
int findSeaPoint(Mat isSkin,Point &start);
void calcRegionVariance(Mat isSkin,Mat variance,Point pos,int &sharpSum,int &validNums);
void calcRegionVariance2(Mat isSkin,Mat variance,Point pos,int &sharpSum,int &validNums);
int findSeaPoint(Mat isSkin,Point &start);
/*
 将多幅图像融合显示
 @para row 子行数
 @para col 子列数
 @para width 融合图像宽度
 @para height 融合图像高度
 @para image  子图像容器
 @para MerImg 融合图像
 @para isResizeImage 1：不缩放子图 0：缩放
 */
void MergeImage(int row,
                int col,
                int width,
                int height,
                vector<Mat>&image,
                Mat &MerImg,
                int isResizeImage=1
                );

/*
 计算图像清晰度
 @para srcG      输入的灰度图像
 @para ratio     平坦区和边缘区清晰度权重:0~1
 return: 计算得到的图像清晰度
 */

struct Portrait
{
    int code;
    int cube;
    Mat img;
    Mat imgG;
    Mat isSkin;
    Mat variance;
    Mat resultImg;
    Rect faceRegion;
    Point marks[5];
    Point s[4];
    Rect marksRegion[10];  //Leye,Reye,nose,mouth，forehead，Lforehead,Rforehead,Sforehead
    double sharpness;
    double sharp[10];
    Portrait(){sharpness=0;}
    friend bool operator <(Portrait p1,Portrait p2){
        return p1.sharpness<p2.sharpness;
    }
    void calcRegion();
    
    void calcSharpness();
};
void reCal(int,void*);

//可调参数
int Ratio=16,Ratio_last=8;
int skinR=6;


vector<string> lists,listsM,listsQ;
vector<Portrait> roll,setM,setQ;
vector<Mat> picOrdered;
priority_queue<Portrait> picLists;
double calcVariance(Mat martrix);
void portraitSharpness(Portrait &temp,Mat srcG);
void generate_edge(Mat srcG,Mat &isSkin,Mat &variance,int cube);
//！每个区域的值后期要加上下限


#define debug 0
#define shownum 1
#define draw 0
int cubeSize=5;
int qr;
int  cannyThre=40;
int changeflag=0;
int main()
{
    //seetaFace初始化
    seeta::FaceDetection detector((MODEL_DIR + "seeta_fd_frontal_v1.0.bin").c_str());
    detector.SetMinFaceSize(40);
    detector.SetScoreThresh(2.f);
    detector.SetImagePyramidScaleFactor(0.8f);
    detector.SetWindowStep(4, 4);
    
    // Initialize face alignment model
    seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());
    
    //读取原图像
    
    
    readLists("../lists.txt",lists,"../Picture/");      //前缀填入: "图片文件夹路径/" 如果图在可执行文件目录下，则不填
    readLists("../../listsM.txt",listsM,"../../data/mohu/");
    readLists("../../listsQ.txt",listsQ,"../../data/qingxi/");
#if debug
    Portrait temp;
    temp.img=imread(lists[66]);  //93 66 110 35 135 114 30
    temp.resultImg=temp.img.clone();
    Mat isSkin,variance;
    cvtColor(temp.img, temp.imgG, CV_BGR2GRAY);
    int result=detectFace(temp.imgG,detector,temp.faceRegion,point_detector,temp.marks);
    if(result!=-1)
    {
        temp.cube=cubeSize;
        temp.calcSharpness();
    }
    cout<<temp.sharpness<<endl;
    //generate_edge(temp.imgG(temp.faceRegion),isSkin,variance,5);
    //     AssessSharpness(temp.img);
    //   // temp.code=77;
    
    //    int result=-1;
    //    result=detectFace(srcG,detector,temp.faceRegion,point_detector,temp.marks);
    //    if(result!=-1)
    //    {
    //        portraitSharpness(temp,srcG);
    //        imshow("result",temp.resultImg);
    //        waitKey(0);
    //    }
    
#else
    //读取图像并计算清晰度
    for(int i=0;i<100;i++)
    {
        Portrait temp;
        temp.code=i;
        temp.img=imread(listsM[i]);  //93 66 110 35 135 114 30
        temp.resultImg=temp.img.clone();
        Mat isSkin,variance;
        cvtColor(temp.img, temp.imgG, CV_BGR2GRAY);
        int result=detectFace(temp.imgG,detector,temp.faceRegion,point_detector,temp.marks);
        if(result!=-1)
        {
            temp.cube=cubeSize;
            temp.calcSharpness();
            // cout<<i<<"sh:"<<temp.sharpness<<endl;
            // imshow("pic"+format("%d",i),temp.img);
            setM.push_back(temp);
        }
        else
        {
            temp.sharpness=0;
            setM.push_back(temp);
        }
    }
    
    
    
    for(int i=0;i<100;i++)
    {
        Portrait temp;
        temp.code=i;
        temp.img=imread(listsQ[i]);  //93 66 110 35 135 114 30
        temp.resultImg=temp.img.clone();
        Mat isSkin,variance;
        cvtColor(temp.img, temp.imgG, CV_BGR2GRAY);
        int result=detectFace(temp.imgG,detector,temp.faceRegion,point_detector,temp.marks);
        if(result!=-1)
        {
            temp.cube=cubeSize;
            temp.calcSharpness();
             //cout<<i<<"sh:"<<temp.sharpness<<endl;
            // imshow("pic"+format("%d",i),temp.img);
            setQ.push_back(temp);
        }
        else
        {
            temp.sharpness=0;
            setQ.push_back(temp);
        }
        
    }
    double accuracy=0;
    for(int i=0;i<100;i++)
    {
        for(int j=0;j<100;j++)
        {
            if(setQ[i].sharpness>=setM[i].sharpness)
                accuracy++;
        }
    }
    accuracy/=100;
    cout<<"accuracy"<<accuracy<<endl;
    //    for(int i=0;i<160;i++
    //    {
    //        Portrait temp;
    //        temp.img=imread(lists[i]);  //93 66 110 35 135
    //        temp.code=i;
    //        Mat srcG;
    //        cvtColor(temp.img, srcG, CV_BGR2GRAY);
    //        int result=-1;
    //        result=detectFace(srcG,detector,temp.faceRegion,point_detector,temp.marks);
    //        if(result!=-1)
    //        {
    //            portraitSharpness(temp,srcG);
    //           // cout<<i<<"sh:"<<temp.sharpness<<endl;
    //           // imshow("pic"+format("%d",i),temp.img);
    //            picLists.push(temp);
    //        }
    //
    //    }
    //    while(!picLists.empty())
    //    {
    //
    //        picOrdered.push_back(picLists.top().resultImg);
    //        roll.push_back(picLists.top());
    //        picLists.pop();
    //    }
    //
    //    //显示
    //    Mat show;
    //    MergeImage(9,18,1300,700,picOrdered,show,1);
    namedWindow("show", CV_WINDOW_AUTOSIZE);
    //imshow("show",show);
    createTrackbar("cubeSize:\n", "show", &cubeSize,  10);
    createTrackbar("cannyThre:\n", "show", &cannyThre, 90);
    createTrackbar("changeflag:\n", "show", &changeflag, 1, reCal);
    
#endif
    waitKey(0);
    return 0;
}

void reCal(int,void* param)
{
    for(int i=0;i<100;i++)
    {
        Portrait &temp=setM[i];
        
        Mat srcG;
        cvtColor(temp.img, srcG, CV_BGR2GRAY);
        
        if(temp.sharpness!=0)
        {
            temp.cube=cubeSize;
            temp.calcSharpness();
            // cout<<i<<"sh:"<<temp.sharpness<<endl;
            // imshow("pic"+format("%d",i),temp.img);
            //   setM.push_back(temp);
        }
        else
        {
            // temp.sharpness=0;
            // setM.push_back(temp);
        }
        
    }
    for(int i=0;i<100;i++)
    {
        Portrait &temp=setQ[i];
        Mat srcG;
        cvtColor(temp.img, srcG, CV_BGR2GRAY);
        
        if(temp.sharpness!=0)
        {
            temp.cube=cubeSize;
            temp.calcSharpness();
            // cout<<i<<"sh:"<<temp.sharpness<<endl;
            // imshow("pic"+format("%d",i),temp.img);
            // setQ.push_back(temp);
        }
        else
        {
            //temp.sharpness=0;
            //setQ.push_back(temp);
        }
        
    }
    double accuracy=0;
    for(int i=0;i<100;i++)
    {
        for(int j=0;j<100;j++)
        {
            if(setQ[i].sharpness>setM[i].sharpness)
                accuracy++;
        }
    }
    accuracy/=10000;
    cout<<"cubeSize:"<<cubeSize<<endl;
    cout<<"cannyThre:"<<cannyThre<<endl;
    cout<<"accuracy:"<<accuracy<<endl;
}

void Portrait::calcSharpness()
{
    double sum=0,v=0;
    calcRegion();
    for(int i=0;i<4;i++)
    {
        generate_edge(imgG(marksRegion[i]),isSkin,variance,cube);
        
        int r=findSeaPoint(isSkin,s[i]);
        int sharpSum=0;
        int validNums=0;
        if(r==-1)
        {
            sharp[i]=0;
            continue;
        }
        calcRegionVariance(isSkin,variance,s[i],sharpSum,validNums);
        if(validNums!=0)
        {
            sharp[i]=1.0*sharpSum;
            sum+=sharp[i];
            v++;
        }
        else
        {
            sharp[i]=0;
        }
    }
    //if(v!=0)
    sum/=4;
    sharpness=sum;
   // cout<<"sharp:"<<sum<<endl;
    for(int i=0;i<4;i++)
    {
        rectangle(resultImg, marksRegion[i], CV_RGB(255, 0, 0));
    }
    imshow("result",resultImg);
}
void generate_edge(Mat srcG,Mat &isSkin,Mat &variance,int cube)
{
    Mat edge;
    Mat result,r2;
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    Canny(srcG,edge,cannyThre,cannyThre*2,3);
    imshow("edge",edge);
    isSkin=Mat::zeros(edge.rows/cube,edge.cols/cube,CV_8UC1);
    result=255*Mat::ones(edge.rows/cube,edge.cols/cube,CV_8UC1);
    variance=Mat::zeros(edge.rows/cube,edge.cols/cube,CV_8UC1);
    for(int i=0;i<edge.rows-cube;i=i+cube)
    {
        for(int j=0;j<edge.cols-cube;j=j+cube)
        {
            
            Rect c(Point(j,i),Point(j+cube,i+cube));
            Scalar s=sum(edge(c));
            if(s[0]==0)
            {
                variance.at<uchar>(i/cube,j/cube)=calcVariance(srcG(c));
                isSkin.at<uchar>(i/cube,j/cube)=255;
                result.at<uchar>(i/cube,j/cube)=0;
            }
            
        }
    }
    //    erode(isSkin, isSkin, element);
    //    dilate(result,result,element);
    //    erode(result,result,element);
    resize(result, result, edge.size());
    imshow("isSkin",result);
    r2=variance*50;
    resize(r2, r2, edge.size());
    imshow("variance",r2);
}
double calcVariance(Mat martrix)
{
    Mat     mean;
    Mat     stddev;
    //    double min,max;
    //    minMaxLoc(martrix, &min, &max);
    //    if(max-min>0.3*max)
    //        return 0;
    meanStdDev ( martrix, mean, stddev);
    // uchar       mean_pxl = mean.val[0];
    double       stddev_pxl = stddev.at<double>(0);
    return stddev_pxl*stddev_pxl;
}

int findSeaPoint(Mat isSkin,Point &start)
{
    static int num=0;
    Mat show=isSkin.clone();
    vector<vector<Point>> contours;
    findContours(isSkin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    //drawContours(show, contours, -1, Scalar(255));
    
    //框出矩形
    int max=0,maxi;
    Rect maxR;
    for (int i = 0; i < contours.size(); i++) {
        int area=contourArea(contours[i]);
        //if (area>0.1*isSkin.rows*isSkin.cols) {
        if(area>max)
        {
            //rectangle(show, bndRect, Scalar(255));
            max=area;
            maxi=i;
        }
        //  }
    }
    if(max==0)
        return -1;
    else
    {
        Moments mu;
        mu= moments( contours[maxi], false );
        start=Point2d( mu.m10/mu.m00 , mu.m01/mu.m00 );
    }
    circle(show, start, 3, Scalar(140),-1);
    Size s( isSkin.rows*4,isSkin.cols*4);
    resize(show, show,s);
#if debug
    imshow("isSkin2"+to_string(num), show);
    num++;
#endif
    return 0;
}

void calcRegionVariance(Mat isSkin,Mat variance,Point pos,int &sharpSum,int &validNums)
{
    if(pos.x<0||pos.x>=isSkin.cols||pos.y<0||pos.y>=isSkin.rows)
        return;
    if(isSkin.at<uchar>(pos.y,pos.x)==0)
        return;
    else
    {
        isSkin.at<uchar>(pos.y,pos.x)=0;
        validNums++;
        sharpSum+=variance.at<uchar>(pos.y,pos.x);
        for(int xx=-1;xx<=1;xx++)
        {
            for(int yy=-1;yy<=1;yy++)
            {
                if(xx==0&&yy==0)
                    continue;
                Point next(pos.x+xx,pos.y+yy);
                calcRegionVariance(isSkin,variance,next,sharpSum,validNums);
            }
        }
    }
    imshow("af",variance);
}

void calcRegionVariance2(Mat isSkin,Mat variance,Point pos,int &sharpSum,int &validNums)
{
    for(int i=0;i<isSkin.rows;i++)
    {
        for(int j=0;j<isSkin.cols;j++)
        {
            if(isSkin.at<uchar>(i,j)==255)
            {
                sharpSum+=variance.at<uchar>(i,j);
                validNums++;
            }
            
        }
    }
}
void Portrait::calcRegion()
{
    marksRegion[0].x=faceRegion.x;
    marksRegion[0].y=faceRegion.y;
    marksRegion[0].width=marks[2].x-faceRegion.x;
    marksRegion[0].height=marks[0].y-faceRegion.y;
    marksRegion[0]=RectChangeBase(img, marksRegion[0]);
    
    marksRegion[1].x=marks[2].x+1;
    marksRegion[1].y=faceRegion.y;
    marksRegion[1].width=faceRegion.x+faceRegion.width-marks[2].x-1;
    marksRegion[1].height=marks[0].y-faceRegion.y;
    marksRegion[1]=RectChangeBase(img, marksRegion[1]);
    
    marksRegion[2].x=faceRegion.x;
    marksRegion[2].y=marks[0].y;
    marksRegion[2].width=marks[2].x-faceRegion.x;
    marksRegion[2].height=faceRegion.y+faceRegion.height-marks[0].y;
    marksRegion[2]=RectChangeBase(img, marksRegion[2]);
    
    marksRegion[3].x=marks[2].x+1;
    marksRegion[3].y=marks[0].y;
    marksRegion[3].width=faceRegion.x+faceRegion.width-marks[2].x-1;
    marksRegion[3].height=faceRegion.y+faceRegion.height-marks[0].y;
    marksRegion[3]=RectChangeBase(img, marksRegion[3]);
    
}

Rect RectChangeBase(const Rect &big,const Rect &small)
{
    Rect newR=small;
    newR.x=small.x-big.x;
    if(newR.x<big.x)
        newR.x=big.x;
    if(newR.x>=big.width)
        newR.x=big.width-1;
    newR.y=small.y-big.y;
    if(newR.y<big.y)
        newR.y=big.y;
    if(newR.y>=big.height)
        newR.y=big.height-1;
    if(newR.width>=big.width-newR.x)
        newR.width=big.width-newR.x-1;
    if(newR.height>=big.height-newR.y)
        newR.height=big.height-newR.y-1;
    return newR;
}

Rect RectChangeBase(const Mat &pic,const Rect &small)
{
    Rect newR=small;
    if(newR.x<0)
        newR.x=0;
    if(newR.x>=pic.cols)
        newR.x=pic.cols-1;
    if(newR.y<0)
        newR.y=0;
    if(newR.y>=pic.rows)
        newR.y=pic.rows-1;
    if(newR.x+newR.width>=pic.cols)
        newR.width=pic.cols-newR.x-1;
    if(newR.y+newR.height>=pic.rows)
        newR.height=pic.rows-newR.y-1;
    return newR;
}





void filiterEdge(Mat ed,Mat & fi)
{
    fi=Mat::zeros(ed.size(),CV_8UC1);
    for (int i = 1; i < ed.rows-1; i++)
    {
        uchar * dataP = ed.ptr<uchar>(i-1);
        uchar * data = ed.ptr<uchar>(i);
        uchar * dataA = ed.ptr<uchar>(i+1);
        for (int j = 1; j < ed.cols * 1-1; j++)
        {
            if(data[j]!=0)
            {
                int sum=dataP[j-1]!=0+dataP[j]!=0+dataP[j+1]!=0+data[j-1]!=0+data[j+1]!=0+dataA[j-1]+dataA[j]!=0+dataA[j+1]!=0;
                if(sum>=2)
                    fi.at<uchar>(i,j)=255;
            }
        }
        
    }
}






int detectFace(Mat imgG,seeta::FaceDetection &detector,Rect &face,seeta::FaceAlignment &point_detector,Point marks[])
{
    //load image
    Mat imgResult=imgG.clone();
    int pts_num = 5;
    int im_width = imgG.cols;
    int im_height = imgG.rows;
    unsigned char* data = new unsigned char[im_width * im_height];
    unsigned char* data_ptr = data;
    unsigned char* image_data_ptr = imgG.ptr<uchar>(0);
    int h = 0;
    for (h = 0; h < im_height; h++) {
        memcpy(data_ptr, image_data_ptr, im_width);
        data_ptr += im_width;
        image_data_ptr = imgG.ptr<uchar>(h+1);
    }
    
    seeta::ImageData image_data;
    image_data.data = data;
    image_data.width = im_width;
    image_data.height = im_height;
    image_data.num_channels = 1;
    
    // Detect faces
    std::vector<seeta::FaceInfo> faces = detector.Detect(image_data);
    int32_t face_num = static_cast<int32_t>(faces.size());
    
    if (face_num == 0)
    {
        delete[]data;
        cout<<"found no face!"<<endl;
        return -1;
    }
    
    // Detect 5 facial landmarks
    seeta::FacialLandmark points[5];
    point_detector.PointDetectLandmarks(image_data, faces[0], points);
    
    // Visualize the results
    Point start(max(faces[0].bbox.x,0), max(faces[0].bbox.y,0));
    Point end(min(faces[0].bbox.x+faces[0].bbox.width - 1,imgG.cols-1),min(faces[0].bbox.y+faces[0].bbox.height - 1,imgG.rows-1));
    face=Rect(start,end);
    rectangle(imgResult, face, CV_RGB(255, 0, 0));
    for (int i = 0; i<pts_num; i++)
    {
        marks[i].x=points[i].x;
        marks[i].y=points[i].y;
        circle(imgResult, marks[i], 4, CV_RGB(255, 255, 0), CV_FILLED);
    }
    imshow("result.jpg", imgResult);
    //AssessSharpness(imgG);
    imwrite("result.jpg", imgResult);
    delete[]data;;
    return 0;
}






int readLists(const char *fileName,vector<string> & lists,string prefix)
{
    FILE * fp;
    string name;
    
    fp=fopen(fileName,"r");
    int flag=1;  //未遇到空格
    do
    {
        char c=fgetc(fp);
        if(flag==1)     //未遇到空格
        {
            
            if(c==' '||c=='\n'||c=='\r'||c=='\t'||feof(fp))
            {
                flag=0;
                lists.push_back(prefix+name);
                //cout<<prefix+name<<endl;
                name.clear();
            }
            else
                name=name+c;
        }
        else
        {
            if(c!=' '&&c!='\n'&&c!='\r'&&c!='\t')
            {
                flag=1;
                name=name+c;
            }
        }
    }while(!feof(fp));
    fclose(fp);
    return lists.size();
}

void MergeImage(int row,int col,int width,int height,vector<Mat>&inputimage,Mat &MerImg,int isResizeImage)
{
    int EachWndWidth;
    int EachWndHeight;
    Rect RectRoi;
    Mat TempImg;
    vector<Mat>image;
    
    //ºÏ≤ÈÕºœÒ◊”¥∞ø⁄∫ÕÕºœÒµƒ ˝¡ø «∑Òœ‡µ»£¨≤ªµ»£∫ ‰≥ˆÃ· æ–≈œ¢
    if (row*col<inputimage.size())
    {
        cout<<"The number of display window is less than number of image and cannot display completely!"<<endl;
    }
    
    
    EachWndWidth=width/col;//√ø∏ˆ◊”¥∞ø⁄µƒøÌ∂»
    EachWndHeight=height/row;//√ø∏ˆ◊”¥∞ø⁄µƒ∏ﬂ∂»
    
    //ºÏ≤ÈÕºœÒµƒÕ®µ¿ ˝£¨Ω´µ•Õ®µ¿ÕºœÒ◊™ªØ≥…»˝Õ®µ¿”√¿¥œ‘ æ
    for (int i=0;i<inputimage.size();i++)
    {
        if (inputimage[i].channels()==1)
        {
            cvtColor(inputimage[i],inputimage[i],CV_GRAY2BGR);
        }
    }
    
    
    // «∑ÒÀı∑≈ÕºœÒ
    if (isResizeImage)
    {
        for (int i=0;i<inputimage.size();i++)
        {
            float    cw=(float)EachWndWidth;    //œ‘ æ«¯µƒøÌ
            float    ch=(float)EachWndHeight;   //œ‘ æ«¯µƒ∏ﬂ
            float    pw=(float)inputimage[i].cols; //‘ÿ»ÎÕºœÒµƒøÌ
            float    ph=(float)inputimage[i].rows;//‘ÿ»ÎÕºœÒµƒ∏ﬂ
            float    cs=cw/ch;//œ‘ æ«¯µƒøÌ∏ﬂ±»
            float    ps=pw/ph;//‘ÿ»ÎÕºœÒµƒøÌ∏ﬂ±»
            float    scale=(cs>ps)?(ch/ph):(cw/pw); //Àı∑≈±»¿˝“Ú◊”
            int      rw=(int)pw*scale;//Àı∑≈∫ÛÕº∆¨µƒøÌ
            int      rh=(int)ph*scale;//Àı∑≈∫ÛÕº∆¨µƒ∏ﬂ
            Mat      TempMat=Mat(Size(rw,rh),CV_8UC3);
            Mat      DisplayMat=Mat(Size(cw,ch),CV_8UC3);
            Rect     RectRoiRoi=Rect((cw-rw)/2,(ch-rh)/2,rw,rh);
            
            resize(inputimage[i],TempMat,Size(rw,rh));
            TempMat.copyTo(DisplayMat(RectRoiRoi));
            image.push_back(DisplayMat);
        }
    }
    else
    {
        for (int i=0;i<inputimage.size();i++)
        {
            image.push_back(inputimage[i]);
        }
    }
    
    //œ‘ æ¥∞ø⁄÷–œ‘ æµƒÕºœÒ
    MerImg=Mat(height,width,CV_8UC3);
    for (int i=0;i<row;i++)
    {
        for (int j=0;j<col;j++)
        {
            RectRoi=Rect(0+j*EachWndWidth,0+i*EachWndHeight,EachWndWidth,EachWndHeight);
            if (i*col+j<image.size())
            {
                resize(image[i*col+j],TempImg,Size(EachWndWidth,EachWndHeight));
                TempImg.copyTo(MerImg(RectRoi));
            }
        }
    }
    image.clear();
}


