"""Quality-gated file enrollment using the canonical InsightFace engine."""
from pathlib import Path
import cv2
from .schemas import FaceDetection

def validate_enrollment_image(path,engine,quality_scorer,min_face_size=30,min_blur_variance=40):
    source=str(Path(path).expanduser())
    image=cv2.imread(source,cv2.IMREAD_COLOR)
    if image is None or image.size==0:return {"accepted":False,"reason":"invalid_or_corrupt_image","source":source}
    faces=engine.detect(image,need_embedding=True)
    if not faces:return {"accepted":False,"reason":"no_face","source":source}
    if len(faces)!=1:return {"accepted":False,"reason":"multiple_faces","source":source}
    raw=faces[0];bbox=tuple(float(v) for v in raw["bbox"]);x1,y1,x2,y2=[int(v) for v in bbox]
    if min(x2-x1,y2-y1)<int(min_face_size):return {"accepted":False,"reason":"face_too_small","source":source}
    crop=image[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]
    if crop.size==0:return {"accepted":False,"reason":"invalid_face_crop","source":source}
    blur=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
    if blur<float(min_blur_variance):return {"accepted":False,"reason":"blur","source":source,"blur":blur}
    detection=FaceDetection(bbox,float(raw.get("det_score",0)),raw.get("landmarks"),raw.get("pose"),raw.get("embedding"))
    quality=quality_scorer.score(image,detection)
    if not quality.accepted:return {"accepted":False,"reason":"quality","source":source,"quality":quality.score}
    value=__import__("numpy").asarray(raw["embedding"],__import__("numpy").float32);norm=float(__import__("numpy").linalg.norm(value))
    if not norm:return {"accepted":False,"reason":"invalid_embedding","source":source}
    return {"accepted":True,"source":source,"embedding":value/norm,"quality":quality.score,"blur":blur}
