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
void Calc_PAV_PSG(Mat img,Mat edge,double &PAV,double &PSG,Rect keyRegion);
Rect RectChangeBase(const Rect &big,const Rect &small);
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
double AssessSharpness(Mat srcG,Rect keyRegion,double ratio=0.6);
struct Portrait
{
    Mat img;
    Mat resultImg;
    Rect faceRegion;
    Point marks[5];
    double sharpness;
    friend bool operator <(Portrait p1,Portrait p2){
        return p1.sharpness<p2.sharpness;
    }
};
void reCal(int,void*);
int Ratio=6,Ratio_last=6;
vector<string> lists;
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
    //    Mat imgG=imread(lists[16],0);
    //    Rect face_area;
    //    detectFace(imgG,detector,face_area,point_detector);
    //
    //    Portrait temp;
    //    Mat face;
    //    int result=-1;
    //    result=detectFace(imgG,detector,face_area,point_detector);
    //    if(result!=-1)
    //        face=imgG(face_area);
    //    else
    //        face=imgG;
    //    temp.img=imgG;
    //    temp.sharpness=AssessSharpness(face,0.6);
    //    putText(temp.img,format("%d",int(temp.sharpness)),Point(50,50),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);
    //    imshow("show",temp.img);
    //
    //读取图像并计算清晰度
    vector<Mat> picOrdered;
    priority_queue<Portrait> picLists;
    // for(int i=0;i<lists.size();i++)
    for(int i=0;i<180;i++)
    {
        Portrait temp;
        temp.img=imread(lists[i]);
        Mat srcG;
        cvtColor(temp.img, srcG, CV_BGR2GRAY);
        
        
        Mat face;
        int result=-1;
        result=detectFace(srcG,detector,temp.faceRegion,point_detector,temp.marks);
        if(result!=-1)
        {
            
            Point start,end;
            start.x=max(temp.faceRegion.x,int(temp.marks[0].x-0.2*temp.faceRegion.width));
            start.y=max(temp.faceRegion.y,int(temp.marks[0].y-0.2*temp.faceRegion.height));
            end.x=min(temp.faceRegion.x+temp.faceRegion.width-1,int(temp.marks[1].x+0.2*temp.faceRegion.width));
            end.y=min(temp.faceRegion.y+temp.faceRegion.height-1,int(temp.marks[4].y+0.2*temp.faceRegion.height));
            Rect face_area(start,end);
            Rect keyR=RectChangeBase(temp.faceRegion, face_area);
            face=srcG(temp.faceRegion);
            temp.sharpness=AssessSharpness(face,keyR,double(Ratio)/10);
            temp.resultImg=temp.img.clone();
//            rectangle(temp.resultImg, face_area, CV_RGB(255, 0, 0));
           putText(temp.resultImg,format("%.2f",temp.sharpness),Point(50,50),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);
//            for (int i = 0; i<5; i++)
//            {
//                circle(temp.resultImg, temp.marks[i], 3, CV_RGB(255, 255, 0), CV_FILLED);
//            }
            
            picLists.push(temp);
        }
    }
    while(!picLists.empty())
    {
        picOrdered.push_back(picLists.top().resultImg);
        picLists.pop();
    }
    
    //显示
    Mat show;
    MergeImage(10,18,1300,700,picOrdered,show,1);
    namedWindow("show", CV_WINDOW_AUTOSIZE);
    imshow("show",show);
    createTrackbar("ratio:\n", "show", &Ratio, 10, reCal,(void *)(&picOrdered));
    Ratio_last=Ratio;
    waitKey(0);
    return 0;
}

void reCal(int,void* param)
{
    if(fabs(Ratio_last-Ratio)>1)
    {
        Ratio_last=Ratio;
    //seetaFace初始化
    seeta::FaceDetection detector((MODEL_DIR + "seeta_fd_frontal_v1.0.bin").c_str());
    detector.SetMinFaceSize(40);
    detector.SetScoreThresh(2.f);
    detector.SetImagePyramidScaleFactor(0.8f);
    detector.SetWindowStep(4, 4);
    
    // Initialize face alignment model
    seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());
    vector<Mat> picOrdered;
    priority_queue<Portrait> picLists;
    // for(int i=0;i<lists.size();i++)
    for(int i=0;i<180;i++)
    {
        Portrait temp;
        temp.img=imread(lists[i]);
        Mat srcG;
        cvtColor(temp.img, srcG, CV_BGR2GRAY);
        
        
        Mat face;
        int result=-1;
        result=detectFace(srcG,detector,temp.faceRegion,point_detector,temp.marks);
        if(result!=-1)
        {
            
            Point start,end;
            start.x=max(temp.faceRegion.x,int(temp.marks[0].x-0.1*temp.faceRegion.width));
            start.y=temp.marks[0].y;
            end.x=min(temp.faceRegion.x+temp.faceRegion.width-1,int(temp.marks[1].x+0.1*temp.faceRegion.width));
            end.y=temp.marks[4].y;
            Rect face_area(start,end);
            Rect keyR=RectChangeBase(temp.faceRegion, face_area);
            face=srcG(temp.faceRegion);
            temp.sharpness=AssessSharpness(face,keyR,double(Ratio)/10);
            temp.resultImg=temp.img.clone();
            //            rectangle(temp.resultImg, face_area, CV_RGB(255, 0, 0));
            putText(temp.resultImg,format("%.2f",temp.sharpness),Point(50,50),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);
            //            for (int i = 0; i<5; i++)
            //            {
            //                circle(temp.resultImg, temp.marks[i], 3, CV_RGB(255, 255, 0), CV_FILLED);
            //            }
            
            picLists.push(temp);
        }
    }
    while(!picLists.empty())
    {
        picOrdered.push_back(picLists.top().resultImg);
        picLists.pop();
    }
    
    //显示
    Mat show;
    MergeImage(10,18,1300,700,picOrdered,show,1);
    imshow("show",show);
    }
}

Rect RectChangeBase(const Rect &big,const Rect &small)
{
    Rect newR=small;
    newR.x=small.x-big.x;
    newR.y=small.y-big.y;
    return newR;
}
double AssessSharpness(Mat srcG,Rect keyRegion,double ratio)
{
    
    Mat kern = (Mat_<double>(3,3) <<
                1, 1,  1,                          // 生成一个掩模核
                1,-8,  1,
                1, 1,  1);
    Mat grad,isEdge,isEdgeF;      //梯度图像,边缘图像
    
    filter2D(srcG, grad, srcG.depth(), kern );  //生成梯度图像
    // imshow("grad",grad);
    double min,max;
    minMaxLoc(grad, &min, &max);
    double thre=(4*min+1*max)/5;
    threshold(grad,isEdge,thre,255,THRESH_BINARY);
    
    //   imshow("isEdge",isEdge);
    filiterEdge(isEdge,isEdgeF);    //滤除边缘杂点
    imshow("isEdgeF",isEdgeF);
    double PAV,PSG;
    Calc_PAV_PSG(srcG,isEdgeF,PAV,PSG,keyRegion);
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
                int sum=dataP[j-1]+dataP[j]+dataP[j+1]+data[j-1]+data[j+1]+dataA[j-1]+dataA[j]+dataA[j+1];
                sum=sum/255;
                if(sum>=2)
                    fi.at<uchar>(i,j)=255;
            }
        }
    }
}

void Calc_PAV_PSG(Mat img,Mat edge,double &PAV,double &PSG,Rect keyRegion)
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
            
            if(keyRegion.contains(Point(j,i)))
            {
                if(isEdge[j]==0)   //平坦区
                {
                    PAV+=(fabs(dataP[j-1]-c)+fabs(dataP[j+1]-c)+fabs(dataA[j-1]-c)+fabs(dataA[j+1]-c))/sqrt(2); //45°方向
                    PAV+=fabs(dataP[j]-c)+fabs(data[j-1]-c)+fabs(data[j+1]-c)+fabs(dataA[j]-c);                //90°方向
                }
                else              //边缘区
                    PSG+=2.5*((data[j+1]-c)*(data[j+1]-c)+(dataA[j]-c)*(dataA[j]-c));
            }
            else
            {
                if(isEdge[j]==0)   //平坦区
                {
                    PAV+=(fabs(dataP[j-1]-c)+fabs(dataP[j+1]-c)+fabs(dataA[j-1]-c)+fabs(dataA[j+1]-c))/sqrt(2); //45°方向
                    PAV+=fabs(dataP[j]-c)+fabs(data[j-1]-c)+fabs(data[j+1]-c)+fabs(dataA[j]-c);                //90°方向
                }
            }
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
            
            if(c==' '||c=='\n'||c=='\r'||feof(fp))
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
            if(c!=' '&&c!='\n'&&c!='\r')
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


