"""Canonical normalized camera recovery ROI contracts."""
from pydantic import BaseModel,Field,field_validator

def _orientation(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def _intersects(a,b,c,d):return _orientation(a,b,c)*_orientation(a,b,d)<0 and _orientation(c,d,a)*_orientation(c,d,b)<0

def validate_polygon(points):
    polygon=[(float(x),float(y)) for x,y in points]
    if len(polygon)<3:raise ValueError("polygon requires at least 3 points")
    if any(not 0.0<=value<=1.0 for point in polygon for value in point):raise ValueError("polygon points must be normalized to 0..1")
    if len(set(polygon))!=len(polygon):raise ValueError("polygon points must be unique")
    area=abs(sum(polygon[i][0]*polygon[(i+1)%len(polygon)][1]-polygon[(i+1)%len(polygon)][0]*polygon[i][1] for i in range(len(polygon))))*.5
    if area<=1e-6:raise ValueError("polygon must have non-zero area")
    count=len(polygon)
    for i in range(count):
        for j in range(i+1,count):
            if j in (i,(i+1)%count) or i==(j+1)%count:continue
            if _intersects(polygon[i],polygon[(i+1)%count],polygon[j],polygon[(j+1)%count]):raise ValueError("polygon must not self-intersect")
    return polygon

class RecoveryROI(BaseModel):
    id:str=Field(pattern=r"^[A-Za-z0-9_-]+$",min_length=1,max_length=64)
    enabled:bool=True
    polygon:list[tuple[float,float]]
    @field_validator("polygon")
    @classmethod
    def valid_polygon(cls,value):return validate_polygon(value)

def unique_rois(value):
    if value is None:return value
    ids=[item.id for item in value]
    if len(ids)!=len(set(ids)):raise ValueError("recovery ROI ids must be unique per camera")
    return value
