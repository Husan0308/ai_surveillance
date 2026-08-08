import cv2,numpy as np
from .schemas import FaceQuality

class FaceQualityScorer:
    def __init__(self,min_score=.55,min_size=24):self.min_score=float(min_score);self.min_size=int(min_size)
    def score(self,image,detection):
        x1,y1,x2,y2=[int(v) for v in detection.bbox];h,w=image.shape[:2]
        complete=float(x1>=0 and y1>=0 and x2<=w and y2<=h);x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
        if x2<=x1 or y2<=y1:return FaceQuality(0,0,0,0,0,complete,False)
        crop=image[y1:y2,x1:x2];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        size=min(1,(min(x2-x1,y2-y1)/max(self.min_size,1))**2);sharp=min(1,float(cv2.Laplacian(gray,cv2.CV_64F).var())/100)
        mean=float(gray.mean());brightness=max(0,1-abs(mean-127)/127)
        pose=1.0
        if detection.pose is not None:
            values=np.abs(np.asarray(detection.pose,float));pose=max(0,1-float(values.max())/45)
        score=.3*size+.25*sharp+.15*brightness+.2*pose+.1*complete
        return FaceQuality(score,size,sharp,brightness,pose,complete,score>=self.min_score)
