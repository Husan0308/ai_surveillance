#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
extern "C" bool NvDsInferParseYolo26Person(const std::vector<NvDsInferLayerInfo>&,
  const NvDsInferNetworkInfo&, const NvDsInferParseDetectionParams&,
  std::vector<NvDsInferObjectDetectionInfo>&);
int main() {
  std::vector<float> tensor(1800, 0.0f);
  NvDsInferLayerInfo layer{}; layer.dataType=FLOAT; layer.buffer=tensor.data();
  layer.inferDims.numDims=2; layer.inferDims.d[0]=300; layer.inferDims.d[1]=6;
  layer.inferDims.numElements=1800;
  NvDsInferNetworkInfo net{640,640,3};
  NvDsInferParseDetectionParams params{}; params.numClassesConfigured=80;
  params.perClassPreclusterThreshold.assign(80,0.25f);
  std::vector<NvDsInferObjectDetectionInfo> out;
  auto parse=[&]() {return NvDsInferParseYolo26Person({layer},net,params,out);};
  auto row=[&](std::initializer_list<float> p) { std::copy(p.begin(),p.end(),tensor.begin()); };
  row({10,20,110,220,.87f,0}); assert(parse() && out.size()==1);
  assert(out[0].classId==0 && out[0].left==10 && out[0].top==20 && out[0].width==100 && out[0].height==200 && out[0].rotation_angle==0);
  for(int cls=1;cls<80;++cls){tensor[5]=float(cls);assert(parse()&&out.empty());}
  row({10,20,110,220,.24f,0});assert(parse()&&out.empty());
  tensor[4]=.25f;assert(parse()&&out.size()==1);
  params.perClassPreclusterThreshold[0]=.5f;assert(parse()&&out.empty());params.perClassPreclusterThreshold[0]=.25f;
  row({-10,-20,700,800,.8f,0});assert(parse()&&out.size()==1&&out[0].left==0&&out[0].top==0&&out[0].width==640&&out[0].height==640);
  for(auto p: {std::vector<float>{20,20,10,30,.8f,0},{10,10,10,20,.8f,0},{700,700,800,800,.8f,0},{10,30,20,10,.8f,0}}){std::copy(p.begin(),p.end(),tensor.begin());assert(parse()&&out.empty());}
  for(float cls: {-1.f,.4f,80.f,1e30f}){row({10,20,100,200,.8f,cls});assert(parse()&&out.empty());}
  for(int i=0;i<6;++i){row({10,20,100,200,.8f,0});tensor[i]=std::numeric_limits<float>::quiet_NaN();assert(parse()&&out.empty());}
  row({10,20,100,200,1.1f,0});assert(parse()&&out.empty());
  row({10,20,110,220,.87f,0});std::copy(tensor.begin(),tensor.begin()+6,tensor.begin()+6);
  assert(parse()&&out.size()==2); // No hidden NMS, even on identical candidates.
  layer.inferDims.d[1]=7;assert(!parse()&&out.empty());layer.inferDims.d[1]=6;
  layer.inferDims.numDims=3;assert(!parse());layer.inferDims.numDims=2;
  layer.inferDims.numElements=1;assert(!parse());layer.inferDims.numElements=1800;
  layer.dataType=HALF;assert(!parse());layer.dataType=FLOAT;
  layer.buffer=nullptr;assert(!parse());layer.buffer=tensor.data();
  assert(!NvDsInferParseYolo26Person({},net,params,out));
  params.perClassPreclusterThreshold.clear();assert(!parse());
  std::puts("PASS: YOLO26 parser shape, person-only, threshold, coordinates, malformed data, zero-init and no-NMS tests");
}
