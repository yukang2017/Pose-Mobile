//
//  common.h
//  Paddle-iOS-Demo
//
//  Created by Sun,Ming(IDL) on 17/12/4.
//  Copyright © 2017年 BaiduIDL. All rights reserved.
//

#ifndef common_h
#define common_h

#include <vector>
#include <map>
using namespace std;
int PostProcessing(const float* layer_scores, vector<vector<vector<float>> > &person,double scale);

// model
//
class ModelDescriptor {
public:
    ModelDescriptor();
    int get_number_parts();
    int number_limb_sequence();
    std::vector<int> &get_limb_sequence();
    std::vector<int> &get_map_idx();
    std::string &get_part_name(int partIndex);
    std::vector<int> mLimbSequence;
private:
    std::map<int, std::string> mPartToName;
    std::vector<int> mMapIdx;
    int mNumberParts;

};


#endif /* common_h */
