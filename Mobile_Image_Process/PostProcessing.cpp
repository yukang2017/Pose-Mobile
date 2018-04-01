//
//  common.cpp
//  Paddle-iOS-Demo
//
//  Created by Sun,Ming(IDL) on 17/12/4.
//  Copyright © 2017年 BaiduIDL. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include "common.h"
#include <vector>


#import "opencv2/core/core.hpp"
#import <opencv2/highgui/highgui.hpp>
#import "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/cap_ios.h"

using namespace std;

int ConnectLimbsCOCO(std::vector< std::vector<double> > &subset,const float* heatmap_pointer,const float *in_peaks,int max_peaks,float *joints);


//input result 57 * w * h
int PostProcessing(const float* layer_scores, vector<vector<vector<float>> > & person,double scale){
    
    //layer_scores[0] refer to the last feature map
    
    
    // using the opencv function
    // heatmap_pointer refer to resized feature map
    //cout <<" layer_scores " << layer_scores[0] << endl;
    int src_c=57,src_h=28,src_w=28;
    int det_c=57,det_h=224,det_w=224;
    float* heatmap_pointer = new float[det_c * det_h * det_w];
    for(int i=0;i<src_c;i++)
    {
        cv::Mat src(src_w,src_h,CV_32FC1);
        int src_offset2 = src_h * src_w;
        //std::vector<float> src_pointer=layer_scores[0];
        for(int x=0;x<src_w;x++)
        {
            for(int y=0;y<src_h;y++)
            {
                src.at<float>(x,y) = layer_scores[i * src_offset2 + y*src_w + x ];
                float tt = layer_scores[i * src_offset2 + y*src_w + x ];
                //if(tt > 0.1)
                //    cout << "result ----" << tt << " ";
                
            }
        }
        //resize
        cv::Mat dst(det_w,det_h,CV_32FC1);
        cv::resize(src,dst,dst.size(),0,0,CV_INTER_CUBIC);
        // fill top
        int dst_offset2 = det_h*det_w;
        for(int x=0;x<det_w;x++)
        {
            for(int y=0;y<det_h;y++)
            {
                heatmap_pointer[i*dst_offset2+y*det_w+x] = dst.at<float>(x,y);
            }
        }
    }
    
    //nms
    float nms_threshold = 0.05;
    int max_peaks = 100;
    const int offset = (max_peaks + 1) * 3;
    float* peaks = new float[18 * (max_peaks + 1) * 3];
    for(int c = 0; c < 18; c++){
        int peakCount = 0;
        for(int h = 0; h < det_h; h++){
            for(int w = 0; w < det_w; w++){
                float value = heatmap_pointer[c * det_h * det_w + h * det_w + w];
                //cout << value << " ";
                if(value < nms_threshold) continue;
                const float top = (h == 0) ? 0 : heatmap_pointer[c * det_h * det_w + (h-1) * det_w + w];
                const float bottom = (h == det_h - 1) ? 0 : heatmap_pointer[c * det_h * det_w + (h+1)*det_w + w];
                const float left = (w == 0) ? 0 : heatmap_pointer[c * det_h * det_w + h*det_w + (w-1)];
                const float right = (w == det_w - 1) ? 0 : heatmap_pointer[c * det_h * det_w  + h*det_w + (w+1)];
                if(value > top && value > bottom && value > left && value > right){
                    peaks[c * offset + (peakCount + 1) * 3] = w;
                    peaks[c * offset + (peakCount + 1) * 3 + 1] = h;
                    //cout << w << " " << h << endl;
                    peaks[c * offset + (peakCount + 1) * 3 + 2] = value;
                    peakCount++;
                }
            }
        }
        //the first ele is total count
        peaks[c * offset] = peakCount;
        cout << "peakCount -----" << peakCount << endl;
        
    }
    //return peaks;
    
    //connect
    std::vector< std::vector<double> > subset;
    int NMS_MAX_PEAKS=100;
    int MAX_NUM_PARTS = 70;
    //int NMS_NUM_PARTS = 18;
    int MAX_PEOPLE=100;
    //get the nnumber of people
    //get joints
    float joints[MAX_NUM_PARTS*3*MAX_PEOPLE];
    int cnt = ConnectLimbsCOCO(subset, heatmap_pointer, peaks, NMS_MAX_PEAKS, joints);
    
    
    cout << "the people number is : -------" << cnt << endl;
    delete [] peaks;
    peaks = NULL;
    delete [] heatmap_pointer;
    heatmap_pointer = NULL;
    //cal
    //double SCALE = 1.0;
    //double scale = 1.0/SCALE;
    float a,b,c;
    //string part[19]={"nose","neck","right_shoulder","right_elbow","right_wrist","left_shoulder","left_elbow","left_wrist","right_hip","right_knee","right_ankle","left_hip","left_knee","left_ankle","right_eye","left_eye","right_ear","left_ear","background"};
    vector<float> location,people,tmps;
    vector<vector<float> > bodyparts;
    //vector<vector<vector<float>> > person;
    float min_x,min_y,max_x,max_y;
    int num_parts = 18;
    float det_y, det_x;
    bool sign;
    float ratio, ch_x;
    float r=0.4;
    for (int ip = 0;ip < cnt; ip++) {
        min_x=10000.0;
        min_y=10000.0;
        max_x=0.0;
        max_y=0.0;
        sign=false;
        bodyparts.clear();
        tmps.clear();
        
        for (int k = 0; k < num_parts; k++) {
            a = scale * joints[ip * num_parts * 3 + k * 3 + 0];
            b = scale * joints[ip * num_parts * 3 + k * 3 + 1];
            c = joints[ip * num_parts * 3 + k * 3 + 2];
            if(c > 0){
                if(a < min_x) min_x = a;
                if(a > max_x) max_x = a;
                if(b < min_y) min_y = b;
                if(b > max_y) max_y = b;
            }
            if(k > 13) continue;
            if((k == 10 || k == 13) and c > 0) sign = true;
            tmps.push_back(a);
            tmps.push_back(b);
            tmps.push_back(k);
            //bodyparts[part[k]]=tmps;
        }
        if(sign==false)
            ratio=0.2;
        else
            ratio=0.1;
        
        bodyparts.push_back(tmps);

        det_y = max_y-min_y;
        det_x = max_x-min_x;
        min_y -= ratio*(det_y);
        max_y += ratio*(det_y);
        min_x -= 0.5*ratio*det_x;
        max_x += 0.5*ratio*det_x;
        if(sign){
            ch_x=((max_y-min_y)*r-(max_x-min_x))*0.5;
            min_x -= ch_x;
            max_x += ch_x;
        }
        if(min_y<0) min_y = 0;
        //if(max_y>h) max_y = h;
        if(min_x<0) min_x = 0;
        //if(max_x>w) max_x = w;
        //left,top,width,height
        location.clear();
        location.push_back(min_x);
        location.push_back(min_y);
        location.push_back(max_x - min_x);
        location.push_back(max_y - min_y);
        //last refer to the bbox info
        bodyparts.push_back(location);
        person.push_back(bodyparts);

    }
    //int cnt = 0;
    return cnt;

}


std::map<int, std::string> createPartToName(std::map<int, std::string> &partToNameBaseLine, std::vector<int> &limbSequence, std::vector<int> &mapIdx)
{
    std::map<int, std::string> partToName = partToNameBaseLine;
    for (int l=0;l<limbSequence.size() / 2;l++) {
        int la = limbSequence[2*l+0];
        int lb = limbSequence[2*l+1];
        int ma = mapIdx[2*l+0];
        int  mb = mapIdx[2*l+1];
        partToName[ma] = partToName[la]+"->"+partToName[lb]+"(X)";
        partToName[mb] = partToName[la]+"->"+partToName[lb]+"(Y)";
    }
    return partToName;
}

int ModelDescriptor::get_number_parts() {
    return mNumberParts;
}
int ModelDescriptor::number_limb_sequence() {
    return mLimbSequence.size() / 2;
}
std::vector<int> &ModelDescriptor::get_limb_sequence() {
    return mLimbSequence;
}
std::vector<int> &ModelDescriptor::get_map_idx() {
    return mMapIdx;
}
std::string &ModelDescriptor::get_part_name(const int partIndex) {
    return mPartToName.at(partIndex);
}

ModelDescriptor::ModelDescriptor()
{
    std::map<int, std::string> partToNameBaseLine;
    partToNameBaseLine.insert(make_pair(0,  "Nose"));
    partToNameBaseLine.insert(make_pair(1,  "Neck"));
    partToNameBaseLine.insert(make_pair(2,  "RShoulder"));
    partToNameBaseLine.insert(make_pair(3,  "RElbow"));
    partToNameBaseLine.insert(make_pair(4,  "RWrist"));
    partToNameBaseLine.insert(make_pair(5,  "LShoulder"));
    partToNameBaseLine.insert(make_pair(6,  "LElbow"));
    partToNameBaseLine.insert(make_pair(7,  "LWrist"));
    partToNameBaseLine.insert(make_pair(8,  "RHip"));
    partToNameBaseLine.insert(make_pair(9,  "RKnee"));
    partToNameBaseLine.insert(make_pair(10, "RAnkle"));
    partToNameBaseLine.insert(make_pair(11, "LHip"));
    partToNameBaseLine.insert(make_pair(12, "LKnee"));
    partToNameBaseLine.insert(make_pair(13, "LAnkle"));
    partToNameBaseLine.insert(make_pair(14, "REye"));
    partToNameBaseLine.insert(make_pair(15, "LEye"));
    partToNameBaseLine.insert(make_pair(16, "REar"));
    partToNameBaseLine.insert(make_pair(17, "LEar"));
    partToNameBaseLine.insert(make_pair(18, "Bkg"));
    std::vector<int> limbSequence;
    
    int tmpnum[]={1,2,  1,5,    2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,     1,11,  11,12,   12,13,  1,0,   0,14,    14,16,  0,15,   15,17,  2,16,   5,17};
    for(int i=0;i<38;i++)
    {
        limbSequence.push_back(tmpnum[i]);
    }
    std::vector<int> mapIdx;
    int tmpnum2[]={31,32, 39,40, 33,34, 35,36, 41,42, 43,44, 19,20, 21,22, 23,24, 25,26, 27,28, 29,30, 47,48, 49,50, 53,54, 51,52, 55,56,37,38, 45,46};
    for(int i=0;i<38;i++)
    {
        mapIdx.push_back(tmpnum2[i]);
    }
    mPartToName = createPartToName(partToNameBaseLine, limbSequence, mapIdx);
    mLimbSequence=limbSequence;
    mMapIdx=mapIdx;
    mNumberParts=(int)partToNameBaseLine.size() - 1;
    if (limbSequence.size() != mMapIdx.size())
        throw std::runtime_error{std::string{"limbSequence.size() should be equal to mMapIdx.size()"}};
}

struct ColumnCompare
{
    bool operator()(const std::vector<double>& lhs,
                    const std::vector<double>& rhs) const
    {
        return lhs[2] > rhs[2];
        //return lhs[0] > rhs[0];
    }
};

int ConnectLimbsCOCO(std::vector< std::vector<double> > &subset,const float* heatmap_pointer,const float *in_peaks,int max_peaks,float *joints) {
    ModelDescriptor model_descriptor;
    const int num_parts = model_descriptor.get_number_parts();
    const std::vector<int> limbSeq = model_descriptor.get_limb_sequence();
    const std::vector<int> mapIdx = model_descriptor.get_map_idx();
    const int  number_limb_seq = model_descriptor.number_limb_sequence();
    int SUBSET_CNT = num_parts+2;
    int SUBSET_SCORE = num_parts+1;
    int SUBSET_SIZE = num_parts+3;
    const int peaks_offset = 3*(max_peaks+1);
    const float *peaks = in_peaks;
    subset.clear();
    int DISPLAY_RESOLUTION_WIDTH=224;
    int DISPLAY_RESOLUTION_HEIGHT=224;
    int NET_RESOLUTION_WIDTH=224;
    int NET_RESOLUTION_HEIGHT=224;
    float connect_inter_threshold=0.050;
    float connect_min_subset_score=0.4;
    int connect_min_subset_cnt=3;
    int MAX_PEOPLE=100;
    int connect_inter_min_above_threshold=9;
    for(int k = 0; k < number_limb_seq; k++) {
        const float* map_x = heatmap_pointer + mapIdx[2*k] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
        const float* map_y = heatmap_pointer + mapIdx[2*k+1] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
        int base_x=mapIdx[2*k] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
        int base_y=mapIdx[2*k+1] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
        const float* candA = peaks + limbSeq[2*k]*peaks_offset;
        const float* candB = peaks + limbSeq[2*k+1]*peaks_offset;
        std::vector< std::vector<double> > connection_k;
        int nA = candA[0];
        int nB = candB[0];
        // add parts into the subset in special case
        if (nA ==0 && nB ==0) {
            continue;
        } else if (nA ==0) {
            for(int i = 1; i <= nB; i++) {
                int num = 0;
                int indexB = limbSeq[2*k+1];
                for(int j = 0; j < subset.size(); j++) {
                    int off = limbSeq[2*k+1]*peaks_offset + i*3 + 2;
                    if (subset[j][indexB] == off) {
                        num = num+1;
                        continue;
                    }
                };
                if (num!=0) {
                } else {
                    std::vector<double> row_vec(SUBSET_SIZE, 0);
                    row_vec[ limbSeq[2*k+1] ] = limbSeq[2*k+1]*peaks_offset + i*3 + 2; //store the index
                    row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
                    row_vec[SUBSET_SCORE] = candB[i*3+2]; //second last number in each row is the total score
                    subset.push_back(row_vec);
                }
            }
            continue;
        } else if (nB ==0) {
            for(int i = 1; i <= nA; i++) {
                int num = 0;
                int indexA = limbSeq[2*k];
                for(int j = 0; j < subset.size(); j++) {
                    int off = limbSeq[2*k]*peaks_offset + i*3 + 2;
                    if (subset[j][indexA] == off) {
                        num = num+1;
                        continue;
                    }
                }
                if (num==0) {
                    std::vector<double> row_vec(SUBSET_SIZE, 0);
                    row_vec[ limbSeq[2*k] ] = limbSeq[2*k]*peaks_offset + i*3 + 2; //store the index
                    row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
                    row_vec[SUBSET_SCORE] = candA[i*3+2]; //second last number in each row is the total score
                    subset.push_back(row_vec);
                } else {
                    //LOG(INFO) << "nB==0 discarded would have added";
                }
            }
            continue;
        }
        std::vector< std::vector<double> > temp;
        const int num_inter = 10;
        for(int i = 1; i <= nA; i++) {
            for(int j = 1; j <= nB; j++) {
                float s_x = candA[i*3];
                float s_y = candA[i*3+1];
                float d_x = candB[j*3] - candA[i*3];
                float d_y = candB[j*3+1] - candA[i*3+1];
                float norm_vec = sqrt( d_x*d_x + d_y*d_y );
                if (norm_vec<1e-6) {
                    // The peaks are coincident. Don't connect them.
                    continue;
                }
                float vec_x = d_x/norm_vec;
                float vec_y = d_y/norm_vec;
                float sum = 0;
                int count = 0;
                for(int lm=0; lm < num_inter; lm++) {
                    int my = round(s_y + lm*d_y/num_inter);
                    int mx = round(s_x + lm*d_x/num_inter);
                    if (mx>=NET_RESOLUTION_WIDTH) {
                        //LOG(ERROR) << "mx " << mx << "out of range";
                        mx = NET_RESOLUTION_WIDTH-1;
                        cout << "mx ---------" << mx << endl;
                    }
                    if (my>=NET_RESOLUTION_HEIGHT) {
                        //LOG(ERROR) << "my " << my << "out of range";
                        my = NET_RESOLUTION_HEIGHT-1;
                        cout << "my ---------" << my << endl;
                    }
                    //cout << "mx ---------" << mx << endl;
                    //cout << "my ---------" << my << endl;
                    //CHECK_GE(mx,0);
                    //CHECK_GE(my,0);
                    int idx = my * NET_RESOLUTION_WIDTH + mx;
                    if (idx>=NET_RESOLUTION_WIDTH*NET_RESOLUTION_HEIGHT || idx<0)
                        std::cout<< "wrong here!!!!!!!!!!!!!!!!!!" << endl;
                    if (base_x+idx>=13760256)
                        std::cout << "xxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;
                    if (base_y+idx>=13760256)
                        std::cout << "yyyyyyyyyyyyyyyyyyyyyyyyyy" << endl;
                    //CDEBUG_LOG("offsize_x:%d, offsize_y:%d",base_x+idx,base_y+idx);
                    float score = (vec_x*map_x[idx] + vec_y*map_y[idx]);
                    //float score = (vec_x*heatmap_pointer[base_x+idx] + vec_y*heatmap_pointer[base_y+idx]);
                    if (score > connect_inter_threshold) {
                        sum = sum + score;
                        count ++;
                    }
                }
                //float score = sum / count; // + std::min((130/dist-1),0.f)
                if (count > connect_inter_min_above_threshold) {//num_inter*0.8) { //thre/2
                    // parts score + cpnnection score
                    std::vector<double> row_vec(4, 0);
                    row_vec[3] = sum/count + candA[i*3+2] + candB[j*3+2]; //score_all
                    row_vec[2] = sum/count;
                    row_vec[0] = i;
                    row_vec[1] = j;
                    temp.push_back(row_vec);
                }
            }
        }
        //** select the top num connection, assuming that each part occur only once
        // sort rows in descending order based on parts + connection score
        if (temp.size() > 0)
            std::sort(temp.begin(), temp.end(), ColumnCompare());
        int num = std::min(nA, nB);
        int cnt = 0;
        std::vector<int> occurA(nA, 0);
        std::vector<int> occurB(nB, 0);
        for(int row =0; row < temp.size(); row++) {
            if (cnt==num) {
                break;
            }
            else{
                int i = int(temp[row][0]);
                int j = int(temp[row][1]);
                float score = temp[row][2];
                if ( occurA[i-1] == 0 && occurB[j-1] == 0 ) { // && score> (1+thre)
                    std::vector<double> row_vec(3, 0);
                    row_vec[0] = limbSeq[2*k]*peaks_offset + i*3 + 2;
                    row_vec[1] = limbSeq[2*k+1]*peaks_offset + j*3 + 2;
                    row_vec[2] = score;
                    connection_k.push_back(row_vec);
                    cnt = cnt+1;
                    occurA[i-1] = 1;
                    occurB[j-1] = 1;
                }
            }
        }
        //** cluster all the joints candidates into subset based on the part connection
        // initialize first body part connection 15&16
        if (k==0) {
            std::vector<double> row_vec(num_parts+3, 0);
            for(int i = 0; i < connection_k.size(); i++) {
                double indexB = connection_k[i][1];
                double indexA = connection_k[i][0];
                row_vec[limbSeq[0]] = indexA;
                row_vec[limbSeq[1]] = indexB;
                row_vec[SUBSET_CNT] = 2;
                // add the score of parts and the connection
                row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
                //LOG(INFO) << "New subset on part " << k << " subsets: " << subset.size();
                subset.push_back(row_vec);
            }
        } else{
            if (connection_k.size()==0) {
                continue;
            }
            // A is already in the subset, find its connection B
            for(int i = 0; i < connection_k.size(); i++) {
                int num = 0;
                double indexA = connection_k[i][0];
                double indexB = connection_k[i][1];
                for(int j = 0; j < subset.size(); j++) {
                    if (subset[j][limbSeq[2*k]] == indexA) {
                        subset[j][limbSeq[2*k+1]] = indexB;
                        num = num+1;
                        subset[j][SUBSET_CNT] = subset[j][SUBSET_CNT] + 1;
                        subset[j][SUBSET_SCORE] = subset[j][SUBSET_SCORE] + peaks[int(indexB)] + connection_k[i][2];
                    }
                }
                // if can not find partA in the subset, create a new subset
                if (num==0) {
                    std::vector<double> row_vec(SUBSET_SIZE, 0);
                    row_vec[limbSeq[2*k]] = indexA;
                    row_vec[limbSeq[2*k+1]] = indexB;
                    row_vec[SUBSET_CNT] = 2;
                    row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
                    subset.push_back(row_vec);
                }
            }
        }
    }
    //** joints by deleteing some rows of subset which has few parts occur
    int cnt = 0;
    for(int i = 0; i < subset.size(); i++) {
        if (subset[i][SUBSET_CNT]<1) {
            //LOG(INFO) << "BAD SUBSET_CNT";
        }
        if (subset[i][SUBSET_CNT]>=connect_min_subset_cnt && (subset[i][SUBSET_SCORE]/subset[i][SUBSET_CNT])>connect_min_subset_score) {
            for(int j = 0; j < num_parts; j++) {
                int idx = int(subset[i][j]);
                if (idx) {
                    joints[cnt*num_parts*3 + j*3 +2] = peaks[idx];
                    joints[cnt*num_parts*3 + j*3 +1] = peaks[idx-1]* DISPLAY_RESOLUTION_HEIGHT/ (float)NET_RESOLUTION_HEIGHT;//(peaks[idx-1] - padh) * ratio_h;
                    joints[cnt*num_parts*3 + j*3] = peaks[idx-2]* DISPLAY_RESOLUTION_WIDTH/ (float)NET_RESOLUTION_WIDTH;//(peaks[idx-2] -padw) * ratio_w;
                    
                } else {
                    joints[cnt*num_parts*3 + j*3 +2] = 0;
                    joints[cnt*num_parts*3 + j*3 +1] = 0;
                    joints[cnt*num_parts*3 + j*3] = 0;
                }
            }
            cnt++;
            if (cnt==MAX_PEOPLE) break;
        }
    }
    //delete model_descriptor;
    //model_descriptor=NULL;
    return cnt;
}
