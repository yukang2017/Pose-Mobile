//
//  ViewController.m
//  Paddle-iOS-Demo
//
//  Created by Zhao Chen on 14/09/2017.
//  Copyright © 2017 BaiduIDL. All rights reserved.
//

#import "ViewController.h"
#import "opencv2/core/core.hpp"
#import <opencv2/highgui/highgui.hpp>
#import "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/cap_ios.h"
#include "classifier.h"
#include <sys/timeb.h>
#include <iostream>
#include "common.h"
#include <opencv2/highgui/ios.h> // OpenCV2
#include <fstream>

using namespace std;

@interface ViewController () <CvVideoCameraDelegate> {
    Classifier* classifier;
}

@property (strong, nonatomic) IBOutlet UIImageView *imageView;
@property (nonatomic,strong) CvVideoCamera *videoCamera;

@end

@implementation ViewController


cv::Mat myUIImageToMat(UIImage *image){
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    return cvMat;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    //Camera
    self.imageView = [[UIImageView alloc]initWithFrame:self.view.frame];
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:self.imageView];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    //    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPresetiFrame960x540;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.rotateVideo = 90;
    self.videoCamera.defaultFPS = 30;
    
    NSString *paddle_file;
    paddle_file = [NSBundle.mainBundle pathForResource:@"pose" ofType:@"paddle"];
    std::string paddle_file_str = std::string([paddle_file UTF8String]);
    
    classifier = new Classifier;
    classifier->init(paddle_file_str);

    
    [self.view insertSubview:self.imageView atIndex:0];
    [self.videoCamera start];
}

- (void)processImage:(cv::Mat &)image
{

    const unsigned char* image_data =  image.data;
    struct timeb startTime , endTime;
    ftime(&startTime);
    
    int width   = image.cols;
    int height  = image.rows;
    int channel = image.channels();
    //ARGB
    unsigned char * image_B= new unsigned char[width*height];
    unsigned char * image_G= new unsigned char[width*height];
    unsigned char * image_R= new unsigned char[width*height];

    for(int i = 0;i< width*height;i++){
        image_B[i] = image_data[4*i];
        image_G[i] = image_data[4*i+1];
        image_R[i] = image_data[4*i+2];
    }

    unsigned char * image_BGR= new unsigned char[width*height*channel];
    //ARGB-BGR
    for(int i = 0;i < width*height;i++){
        image_BGR[i*3+0] = image_B[i];
        image_BGR[i*3+1] = image_G[i];
        image_BGR[i*3+2] = image_R[i];
    }
    
    cv::Mat imageBGR(image.rows,image.cols,CV_8UC3, cv::Scalar(0,0,0));
    
    cv::cvtColor(image, imageBGR, cv::COLOR_BGRA2BGR);
    cv::resize(imageBGR, imageBGR, cv::Size(224,224),cv::INTER_CUBIC);
    

    const float *result = classifier->process(imageBGR.data, 224, 224 ,224, 224, 1.0);//image.cols, image.rows

    vector<vector<vector<float>> >  person;
    double scale = 1.0;
    int cnt = PostProcessing(result, person,scale);

    cv::vector<cv::Point> Points;
    if(person.size() > 0){
        //cout << "people left ------------------: " << person[0][1][0] << endl;
        // nms part_num * (max_peaks + 1)  * 3
        //cnt by calling
        ModelDescriptor *model_descriptor = new ModelDescriptor();
        std::vector<int> LimbSquence = model_descriptor->mLimbSequence;
        cout << "the peopel number ------------------" << cnt <<  endl;
        int num_points = 17;
        //draw points
        for(int i = 0; i < cnt;i++){
            Points.clear();
            for(int j = 0; j < num_points; j++){
                Points.push_back(cv::Point(person[i][0][3*j]*image.cols/224.0,person[i][0][3*j+1]*image.rows/224.0));
                //printf("Person %d, Points %d :(%d,%d) \n",i,j,Points[j].x,Points[j].y);
                if(Points[j].x*Points[j].y!=0)
                    cv::circle(image, Points[j], 5, cv::Scalar(255, 0, 0), - 1);
            }
            
            if(Points.size()>0){
                float thred = 0.1;
                for(int i = 0; i < LimbSquence.size()/2; i++){
                    if((Points[LimbSquence[2*i]].x < image.cols*thred||
                        Points[LimbSquence[2*i]].y < image.rows*thred||
                        Points[LimbSquence[2*i+1]].x < image.cols*thred||
                        Points[LimbSquence[2*i+1]].y < image.rows*thred))
                        continue;
                    cv::line(image, Points[LimbSquence[2*i]], Points[LimbSquence[2*i+1]], cv::Scalar(0, 255, 0), 1);
                }
            }

        }
    }
    
    
    ftime(&endTime);
    std::cout << std::endl <<"run time: "<<(endTime.time-startTime.time)*1000 + (endTime.millitm - startTime.millitm) << "ms"<<std::endl;

    self.imageView.image = MatToUIImage(image);
}

@end
