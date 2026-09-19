// Raw one-to-many (84,8400) COCO parser. Only Gst-nvinfer performs NMS.
#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
static std::atomic<unsigned long long> errors{0}, rejected{0}, calls{0}, proposals{0};
extern "C" unsigned long long Yolo26ParserErrors(){return errors.load();}
extern "C" unsigned long long Yolo26ParserRejected(){return rejected.load();}
extern "C" unsigned long long Yolo26ParserCalls(){return calls.load();}
extern "C" unsigned long long Yolo26ParserProposals(){return proposals.load();}
static bool invalid(const char *why){
  if(errors.fetch_add(1)==0)std::fprintf(stderr,"YOLO PARSER_ERROR %s\n",why);
  return false;
}
extern "C" bool NvDsInferParseYolo26RawPerson(
 const std::vector<NvDsInferLayerInfo>& layers,const NvDsInferNetworkInfo& net,
 const NvDsInferParseDetectionParams& params,std::vector<NvDsInferObjectDetectionInfo>& objects){
  objects.clear();++calls;
  if(layers.size()!=1 || !layers[0].buffer || layers[0].isInput || layers[0].dataType!=FLOAT ||
     net.width!=640 || net.height!=640 || net.channels!=3 || params.numClassesConfigured!=80 ||
     params.perClassPreclusterThreshold.size()!=80)return invalid("invalid raw layer/network/classes");
  const auto &dim=layers[0].inferDims;
  // DeepStream owns batching and calls the parser on each frame's tensor slice.
  if(dim.numDims!=2 || dim.d[0]!=84 || dim.d[1]!=8400 || dim.numElements!=84*8400)
    return invalid("expected per-frame FLOAT (84,8400)");
  const float threshold=params.perClassPreclusterThreshold[0];
  if(!std::isfinite(threshold)||threshold<0||threshold>1)return invalid("invalid threshold");
  constexpr int anchors=8400;
  const float *data=static_cast<const float*>(layers[0].buffer);
  for(int a=0;a<anchors;++a){
    float score=-1;int cls=-1;bool good=true;
    for(int c=0;c<80;++c){
      const float s=data[(4+c)*anchors+a];
      if(!std::isfinite(s)||s<0||s>1){good=false;break;}
      if(s>score){score=s;cls=c;}
    }
    if(!good){++rejected;continue;}
    if(cls!=0 || score<threshold)continue;
    const float cx=data[a],cy=data[anchors+a],w=data[2*anchors+a],h=data[3*anchors+a];
    if(!std::isfinite(cx)||!std::isfinite(cy)||!std::isfinite(w)||!std::isfinite(h)||w<=0||h<=0){++rejected;continue;}
    // Double intermediates avoid overflow before bounded conversion to DS floats.
    const float x1=std::clamp(double(cx)-double(w)/2,0.0,double(net.width));
    const float y1=std::clamp(double(cy)-double(h)/2,0.0,double(net.height));
    const float x2=std::clamp(double(cx)+double(w)/2,0.0,double(net.width));
    const float y2=std::clamp(double(cy)+double(h)/2,0.0,double(net.height));
    if(x2<=x1||y2<=y1){++rejected;continue;}
    NvDsInferObjectDetectionInfo obj{};
    obj.classId=0;obj.left=x1;obj.top=y1;obj.width=x2-x1;obj.height=y2-y1;obj.detectionConfidence=score;
    objects.push_back(obj); // No sorting, IoU filtering or NMS here.
  }
  proposals+=objects.size();
  return true;
}
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26RawPerson);
