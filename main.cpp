#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <queue>
#include <iostream>
#include <cmath>

#include "face_detection.h"
#include "face_alignment.h"
#define debug 0
#define shownum 1
#define draw 0
using namespace cv;
using namespace std;
//#define MAX(a,b) ( ((a)>(b)) ? (a):(b) )
//#define MIN(a,b) ( ((a)>(b)) ? (b):(a) )
string MODEL_DIR = "../../model/";
int detectFace(Mat imgG,seeta::FaceDetection &detector,Rect &face,seeta::FaceAlignment &point_detector,Point marks[]);

int readLists(const char *  fileName,vector<string> & lists,string prefix="");
void filiterEdge(Mat  ed,Mat & fi);
void Calc_PAV_PSG(Mat img,Mat edge,double &PAV,double &PSG);
Rect RectChangeBase(const Rect &big,const Rect &small);
Rect RectChangeBase(const Mat &pic,const Rect &small);
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
double AssessSharpness(Mat srcG,double ratio=0.8);
struct Portrait
{
    int code;
    Mat img;
    Mat resultImg;
    Rect faceRegion;
    Point marks[5];
    Rect marksRegion[10];  //Leye,Reye,nose,mouth，forehead，Lforehead,Rforehead,Sforehead
    double sharpness;
    double sharp[10];
    Portrait(){sharpness=0;}
    friend bool operator <(Portrait p1,Portrait p2){
        return p1.sharpness<p2.sharpness;
    }
    void calcRegion();
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
//！每个区域的值后期要加上下限
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
    temp.img=imread(lists[93]);  //93 66 110 35 135 114 30
     AssessSharpness(temp.img);
   // temp.code=77;
    Mat srcG;
    cvtColor(temp.img, srcG, CV_BGR2GRAY);
    int result=-1;
    result=detectFace(srcG,detector,temp.faceRegion,point_detector,temp.marks);
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
            temp.img=imread(listsM[i]);  //93 66 110 35 135
            temp.code=i;
            Mat srcG;
            cvtColor(temp.img, srcG, CV_BGR2GRAY);
            int result=-1;
            result=detectFace(srcG,detector,temp.faceRegion,point_detector,temp.marks);
            if(result!=-1)
            {
                portraitSharpness(temp,srcG);
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
        temp.img=imread(listsQ[i]);  //93 66 110 35 135
        temp.code=i;
        Mat srcG;
        cvtColor(temp.img, srcG, CV_BGR2GRAY);
        int result=-1;
        result=detectFace(srcG,detector,temp.faceRegion,point_detector,temp.marks);
        if(result!=-1)
        {
            portraitSharpness(temp,srcG);
            // cout<<i<<"sh:"<<temp.sharpness<<endl;
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
            if(setQ[i].sharpness>setM[i].sharpness)
                accuracy++;
        }
    }
    accuracy/=10000;
    cout<<"accuracy"<<accuracy<<endl;
//    for(int i=0;i<160;i++)
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
    createTrackbar("ratio:\n", "show", &Ratio, 20, reCal,(void *)(&picOrdered));
    createTrackbar("skinR:\n", "show", &skinR, 20, reCal,(void *)(&picOrdered));
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
            portraitSharpness(temp,srcG);
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
            portraitSharpness(temp,srcG);
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
    cout<<"accuracy"<<accuracy<<endl;
//    picOrdered.clear();
//    for(int i=0;i<roll.size();i++)
//    {
//        Portrait temp=roll[i];
//        Mat srcG;
//        cvtColor(temp.img, srcG, CV_BGR2GRAY);
//        portraitSharpness(temp,srcG);
//        picLists.push(temp);
//    }
//    while(!picLists.empty())
//    {
//
//        picOrdered.push_back(picLists.top().resultImg);
//
//        picLists.pop();
//    }
//
//    //显示
//    Mat show;
//    MergeImage(9,18,1300,700,picOrdered,show,1);
//    namedWindow("show", CV_WINDOW_AUTOSIZE);
//    imshow("show",show);
}

void portraitSharpness(Portrait &temp,Mat srcG)
{
    temp.calcRegion();
    temp.resultImg=temp.img.clone();
    priority_queue<double> skin;
    for (int i = 0; i<10; i++)
    {
#if draw
        if(i>=4)
        rectangle(temp.resultImg, temp.marksRegion[i], CV_RGB(255, 0, 0));
#endif
        if(i==0||i==1||i==3)
        {
            //cout<<i<<endl;
            temp.sharp[i]=AssessSharpness(srcG(temp.marksRegion[i]),double(Ratio)/20);
        }
        else if(i>=4)
        {
           // cout<<i<<endl;
            temp.sharp[i]=calcVariance(srcG(temp.marksRegion[i]));
          //  temp.sharp[i]=AssessSharpness(srcG(temp.marksRegion[i]),1);
            if(temp.sharp[i]!=0)
                skin.push(-temp.sharp[i]);
        }
    }
    //左右眼清晰度
    if(temp.sharp[0]-temp.sharp[1]>32)
        temp.sharp[0]=0;
    else if(temp.sharp[1]-temp.sharp[0]>32)
        temp.sharp[1]=0;
    else
    {
        temp.sharp[0]=min(temp.sharp[0],temp.sharp[1]);
        temp.sharp[1]=temp.sharp[0];
    }
    double fSh=0;
    if(skin.size()>=2)
    {
      //skin.pop();
       fSh=-skin.top();
    }

    if(fSh>28)
        fSh=0;
    fSh=min(85.0,fSh);
    temp.sharpness=temp.sharp[0]+temp.sharp[1]+temp.sharp[3]+fSh*skinR/2;
    rectangle(temp.resultImg, temp.faceRegion, CV_RGB(255, 0, 0));
#if shownum
 putText(temp.resultImg,format("%d",temp.code),Point(50,30),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,0),4,8);
 putText(temp.resultImg,format("%d",int(temp.sharpness)),Point(20,30),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);
#endif
    //imshow("Result",temp.resultImg);
    //cout<<temp.sharpness<<endl;
}
double calcVariance(Mat martrix)
{
    Mat     mean;
    Mat     stddev;
    double min,max;
    minMaxLoc(martrix, &min, &max);
    if(max-min>0.3*max)
        return 0;
    meanStdDev ( martrix, mean, stddev);
    // uchar       mean_pxl = mean.val[0];
    double       stddev_pxl = stddev.at<double>(0);
    return stddev_pxl*stddev_pxl;
}

void Portrait::calcRegion()
{
    marksRegion[0].x=marks[0].x-0.1*faceRegion.width;
    marksRegion[0].y=marks[0].y-0.05*faceRegion.height;
    marksRegion[0].width=0.2*faceRegion.width;
    marksRegion[0].height=0.15*faceRegion.height;
    marksRegion[0]=RectChangeBase(img,marksRegion[0]);
    marksRegion[1].x=marks[1].x-0.1*faceRegion.width;
    marksRegion[1].y=marks[1].y-0.05*faceRegion.height;
    marksRegion[1].width=0.2*faceRegion.width;
    marksRegion[1].height=0.15*faceRegion.height;
    marksRegion[1]=RectChangeBase(img,marksRegion[1]);
    
    marksRegion[2].x=marks[2].x-0.1*faceRegion.width;
    marksRegion[2].y=marks[2].y-0.2*faceRegion.height;
    marksRegion[2].width=0.2*faceRegion.width;
    marksRegion[2].height=0.30*faceRegion.height;
    marksRegion[2]=RectChangeBase(img,marksRegion[2]);
    
    //mouth
    int mWidth=marks[4].x-marks[3].x;
    marksRegion[3].x=marks[3].x-0.2*mWidth;
    marksRegion[3].y=marks[3].y-0.1*faceRegion.height;
    marksRegion[3].width=1.4*mWidth;
    marksRegion[3].height=0.25*faceRegion.height;
    marksRegion[3]=RectChangeBase(img,marksRegion[3]);
    
    //forehead
    int eyeY=min(marksRegion[0].y,marksRegion[1].y);
    int upY=faceRegion.y;
    int eyeDis=marksRegion[1].x-marksRegion[0].x;
    marksRegion[4].height=(eyeY-upY)*0.5;
    marksRegion[4].width=eyeDis*0.50;
    marksRegion[4].x=(marksRegion[1].x+marksRegion[0].x+marksRegion[1].width)/2-0.5*marksRegion[4].width;
    marksRegion[4].y=upY+0.2*(eyeY-upY);
    marksRegion[4]=RectChangeBase(img,marksRegion[4]);
    
    int cube=0.5*mWidth;
    marksRegion[5].width=cube;
    marksRegion[5].height=cube;
    marksRegion[5].x=marks[0].x;
    marksRegion[5].y=marksRegion[4].y-marksRegion[5].height*0.3;
    marksRegion[5]=RectChangeBase(img,marksRegion[5]);
    
    marksRegion[6].width=cube;
    marksRegion[6].height=cube;
    marksRegion[6].x=marks[1].x-cube;
    marksRegion[6].y=marksRegion[4].y-marksRegion[6].height*0.3;
    marksRegion[6]=RectChangeBase(img,marksRegion[6]);
    

    
    marksRegion[7].width=cube;
    marksRegion[7].height=cube;
    marksRegion[7].x=(marks[0].x+marks[1].x)/2-cube/2;
    marksRegion[7].y=marksRegion[4].y+marksRegion[7].height*0.2;
    marksRegion[7]=RectChangeBase(img,marksRegion[7]);
    

    marksRegion[8].width=cube;
    marksRegion[8].height=marksRegion[8].width;
    marksRegion[8].x=marks[3].x-marksRegion[8].width;
    marksRegion[8].y=marks[3].y-1.2*marksRegion[8].height;
    marksRegion[8]=RectChangeBase(img,marksRegion[8]);
    
    marksRegion[9].width=cube;
    marksRegion[9].height=marksRegion[9].width;
    marksRegion[9].x=marks[4].x+marksRegion[9].height*0.2;
    marksRegion[9].y=marks[3].y-1.2*marksRegion[9].height;
    marksRegion[9]=RectChangeBase(img,marksRegion[9]);
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

double AssessSharpness(Mat srcG,double ratio)
{
//    Mat     mean;
//    Mat     stddev;
//    double min,max;
//    minMaxLoc(srcG, &min, &max);
//    meanStdDev ( srcG, mean, stddev);
//    double       mean_pxl = sqrt(mean.at<double>(0));
//    if(mean_pxl<10)
//        return 0;
    static int num=0;
    imshow("src"+format("%d",num),srcG);
    Mat kern = (Mat_<double>(3,3) <<
                1, 1,  1,                          // 生成一个掩模核
                1,-8,  1,
                1, 1,  1);
    Mat grad,isEdge,isEdgeF;      //梯度图像,边缘图像
    
    //    filter2D(srcG, grad, srcG.depth(), kern );  //生成梯度图像
    //    // imshow("grad"+format("%d",num),grad);
    //    double min,max;
    //    minMaxLoc(grad, &min, &max);
   // blur(srcG,grad,Size(3,3));
    Canny(srcG,isEdge,25,50,3);
    
    //   imshow("isEdge"+format("%d",num),isEdge);
    //filiterEdge(isEdge,isEdgeF);    //滤除边缘杂点
    imshow("isEdge"+format("%d",num),isEdge);
    //num++;
    double PAV,PSG;
    Calc_PAV_PSG(srcG,isEdge,PAV,PSG);
    //cout<<PAV<<" "<<PSG<<endl;
    

    return ratio*PAV+(1-ratio)*PSG;
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

void Calc_PAV_PSG(Mat img,Mat edge,double &PAV,double &PSG)
{
    PAV=0;
    PSG=0;
    Mat imgT;
    normalize(img, imgT, 0, 100, NORM_MINMAX, -1, Mat());
    
    for (int i = 1; i < img.rows-1; i++)
    {
        
        for (int j = 1; j < img.cols * 1-1; j++)
        {
            uchar * dataP = imgT.ptr<uchar>(i-1);
            uchar * data = imgT.ptr<uchar>(i);
            uchar * dataA = imgT.ptr<uchar>(i+1);
            uchar * isEdge= edge.ptr<uchar>(i);
            
            uchar c=data[j];
            
            if(isEdge[j]==0)   //平坦区
            {
                PAV+=(fabs(dataP[j-1]-c)+fabs(dataP[j+1]-c)+fabs(dataA[j-1]-c)+fabs(dataA[j+1]-c))/sqrt(2); //45°方向
                PAV+=fabs(dataP[j]-c)+fabs(data[j-1]-c)+fabs(data[j+1]-c)+fabs(dataA[j]-c);                //90°方向
            }
            else              //边缘区
                PSG+=((data[j+1]-c)*(data[j+1]-c)+(dataA[j]-c)*(dataA[j]-c));
            
        }
    }
    PAV/=(img.rows*img.cols);
    PSG/=(img.rows*img.cols);
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


