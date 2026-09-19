#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
extern "C" bool NvDsInferParseYolo26RawPerson(const std::vector<NvDsInferLayerInfo>&,
 const NvDsInferNetworkInfo&,const NvDsInferParseDetectionParams&,std::vector<NvDsInferObjectDetectionInfo>&);
int main(){
 constexpr int A=8400;std::vector<float> data(84*A,0);
 NvDsInferLayerInfo layer{};layer.dataType=FLOAT;layer.buffer=data.data();layer.inferDims.numDims=2;
 layer.inferDims.d[0]=84;layer.inferDims.d[1]=A;layer.inferDims.numElements=84*A;
 NvDsInferNetworkInfo net{640,640,3};NvDsInferParseDetectionParams params{};
 params.numClassesConfigured=80;params.perClassPreclusterThreshold.assign(80,.25f);
 std::vector<NvDsInferObjectDetectionInfo> out;
 auto parse=[&](){return NvDsInferParseYolo26RawPerson({layer},net,params,out);};
 auto box=[&](float x,float y,float w,float h,float score){std::fill(data.begin(),data.end(),0);data[0]=x;data[A]=y;data[2*A]=w;data[3*A]=h;data[4*A]=score;};
 box(100,150,80,100,.8f);assert(parse()&&out.size()==1);
 assert(out[0].classId==0&&out[0].left==60&&out[0].top==100&&out[0].width==80&&out[0].height==100&&out[0].rotation_angle==0);
 for(int cls=1;cls<80;++cls){box(100,150,80,100,.8f);data[(4+cls)*A]=.9f;assert(parse()&&out.empty());}
 box(100,150,80,100,.249f);assert(parse()&&out.empty());data[4*A]=.25f;assert(parse()&&out.size()==1);
 params.perClassPreclusterThreshold[0]=.3f;assert(parse()&&out.empty());params.perClassPreclusterThreshold[0]=.25f;
 box(320,320,800,1000,.8f);assert(parse()&&out.size()==1&&out[0].left==0&&out[0].top==0&&out[0].width==640&&out[0].height==640);
 for(float w:{0.f,-1.f}){box(100,100,w,10,.8f);assert(parse()&&out.empty());}
 box(900,900,20,20,.8f);assert(parse()&&out.empty());
 for(int field:{0,1,2,3,4,79})for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){
  box(100,100,20,20,.8f);data[field*A]=bad;assert(parse()&&out.empty());}
 box(100,100,20,20,1.1f);assert(parse()&&out.empty());
 box(100,100,20,20,.8f);for(int ch=0;ch<84;++ch)data[ch*A+1]=data[ch*A];assert(parse()&&out.size()==2); // NMS must happen downstream.
 layer.inferDims.d[0]=300;assert(!parse()&&out.empty());layer.inferDims.d[0]=84;
 layer.inferDims.d[1]=8399;assert(!parse());layer.inferDims.d[1]=A;
 layer.inferDims.numDims=3;assert(!parse());layer.inferDims.numDims=2;
 layer.inferDims.numElements=1;assert(!parse());layer.inferDims.numElements=84*A;
 layer.dataType=HALF;assert(!parse());layer.dataType=FLOAT;
 layer.buffer=nullptr;assert(!parse());layer.buffer=data.data();
 params.perClassPreclusterThreshold.clear();assert(!parse());
 std::puts("PASS: raw parser shape/person/non-person/cxcywh/threshold/NaN/Inf/clamping/invalid boxes/no-NMS tests");
}
