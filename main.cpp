#include<opencv2/opencv.hpp>
#include<apriltag/apriltag.h>
#include<apriltag/tag36h11.h>
#include<vector>
#include<math.h>
#define s 0.084
using namespace std;
using namespace cv;
int hmin=0, smin=0, vmin =0 ;
int hmax=19, smax=240, vmax =255;
Mat HSVframe,
    dst;
//识别颜色数据
vector<vector<int>>myColor{ 
        //{0, 10,43,255, 46,255},//红
        //{156, 180,43,255, 46,255}
        //{0,360,0,100,90,100}//白
      };
//创建点集Point
void getContours(Mat& dilimg,Mat& img) {

	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;

	//寻找轮廓
	findContours(dilimg, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_TC89_L1, Point(0, 0));

	vector<vector<Point>>approxCurve(contours.size());
	
	vector<Rect>boundRect(contours.size());
	vector<Rect>boundRects(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		//cout << area << endl;
		//排除误差
		if (area > 990.52 && area < 10000)
		{
			double pir = arcLength(contours[i], true);
			approxPolyDP(contours[i], approxCurve[i], 0.02 * pir, true);//计算轮廓周长

			//cout << contours[i].size() << endl;
			boundRect[i] = boundingRect(approxCurve[i]);
			boundRects[i] = boundingRect(contours[i]);

			//用矩形框起目标图形
			rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 0), 5);
		}
	}
}
//识别颜色
void getColor(Mat& frame){
	
	//RGB转HSV
	cvtColor(frame, HSVframe, COLOR_RGB2HSV);
	    
	for (int i = 0; i < myColor.size(); i++){

		Scalar min(myColor[i][0], myColor[i][2], myColor[i][4]);
		Scalar max(myColor[i][1], myColor[i][3], myColor[i][5]);
		inRange(HSVframe, min, max, dst);
		getContours(dst,frame);
	}
}
//调节开关
void Track(){
   
	//调节开关(HSV)
	namedWindow("ds", (640, 250));
	createTrackbar("H min", "ds", &hmin, 179);
	createTrackbar("H max", "ds", &hmax, 179);		
	createTrackbar("S min", "ds", &smin, 255);
	createTrackbar("S max", "ds", &smax, 255);
	createTrackbar("V min", "ds", &vmin, 255);
	createTrackbar("V max", "ds", &vmax, 255);
}
//获取识别颜色数据库
 void checkColor(Mat& frame){
    
	cvtColor(frame, HSVframe, COLOR_RGB2HSV);
	//颜色阈值
	Scalar min(hmin,smin,vmin);//没有等于；错误写法：Scalar min=（233，34，35）；
	Scalar max(hmax, smax, vmax);
	inRange(HSVframe, min, max, dst);
	namedWindow("lv tu", WINDOW_NORMAL);
	imshow("lv tu", dst);
    //cout << hmin <<"  " << hmax <<"   " << smin <<"  " << smax <<"  " << vmin <<"  " << vmax << endl;
 }
// 获得纵深坐标
void Detect_AprilTag(Mat& frame,zarray_t *detections,Mat& cameraMatrix,Mat& distCoeffs){
    
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);

            putText(frame, to_string(det->id), Point(det->c[0],det->c[1]), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 1);
            
            //提取角点坐标
            vector<Point2f>corners(4);
            for (int j = 0; j < 4; j++) {
                
                corners[j]=Point(det->p[j][0],det->p[j][1]);
                putText(frame, to_string(j), Point(det->p[j][0],det->p[j][1]), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2);
                int next = (j + 1) % 4;
                cv::line(frame, Point(det->p[j][0],det->p[j][1]), Point(det->p[next][0],det->p[next][1]), cv::Scalar(0, 255, 0), 2);
            }

            //计算单应性矩阵H
            vector<Point2f>objPoints={
                {-s/2,-s/2},
                { s/2,-s/2},
                { s/2, s/2},
                {-s/2, s/2}
            };
            Mat H=findHomography(objPoints,corners);

            //计算3D姿态
            vector<Point3f>objPoints3D={
                {-s/2,-s/2,0},
                { s/2,-s/2,0},
                { s/2, s/2,0},
                {-s/2, s/2,0}
            };
            Mat rvec ,tvec;
            solvePnP(objPoints3D,corners,cameraMatrix,distCoeffs,rvec,tvec);
            
            //获取距离信息
            //直接获取Z轴深度
            double depth=tvec.at<double>(2);
            putText(frame,"depth"+to_string(depth)+"m",Point(frame.rows/2,frame.cols/4),FONT_HERSHEY_SIMPLEX,1.0,Scalar(0,255,0),2);
             
            //计算欧式距离
            double tx=tvec.at<double>(0);
            double ty=tvec.at<double>(1);
            double tz=tvec.at<double>(2);
            double distance=sqrt(tx*tx+ty*ty+tz*tz);
            putText(frame,"distance"+to_string(distance)+"m",Point(frame.rows/4,frame.cols/2),FONT_HERSHEY_SIMPLEX,1.0,Scalar(0,255,0),2);            
            apriltag_detection_destroy(det);
        }
    }
int main(){                         
    
    //相机内参
    Mat cameraMatrix =(Mat_<double>(3,3)<<
    1042.663213393477,0,957.3654775954664,
    0,1074.941559575904,561.3271779981716,
    0,0,1
    );
    
    //畸变系数
    Mat distCoeffs=(Mat_<double>(1,5)<<
    -0.4478107247271904,0.409417167157276,0.00402497631401018,-0.004142017549736123,-0.2837583006667866
    );

    //初始化AprilTag检测器
    apriltag_family_t* tf=tag36h11_create();
    apriltag_detector* td=apriltag_detector_create();
    apriltag_detector_add_family(td,tf);
    td->quad_decimate = 2.0;//降低图像分辨率
    td->quad_sigma = 0.0;//高斯模糊参数

    //读取视频
    VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    
    Mat frame, frame_gray;
    //Track();//调节阈值开关
    while (1) 
    {
        
        bool isRead = cap.read(frame);
        if (!isRead) { break; }
        //瞄准点
        line(frame,Point(frame.cols/2-20,frame.rows/2),Point(frame.cols/2+20,frame.rows/2),Scalar(0,0,255),2,LINE_8);
        line(frame,Point(frame.cols/2,frame.rows/2-20),Point(frame.cols/2,frame.rows/2+20),Scalar(0,0,255),2,LINE_8);
        //checkColor(frame);//获取识别的颜色
        getColor(frame);//颜色识别
        
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // 检测标签
        image_u8_t im = {.width = frame_gray.cols, .height = frame_gray.rows, .stride = frame_gray.cols, .buf = frame_gray.data};
        zarray_t *detections = apriltag_detector_detect(td, &im);

        Detect_AprilTag(frame,detections,cameraMatrix,distCoeffs);//显示检测结果

        imshow("frame", frame);
        waitKey(30);
        zarray_destroy(detections);
    }
    
    // 循环结束后销毁检测器和家族
    apriltag_detector_destroy(td);
    tag36h11_destroy(tf);
    return 0;

}