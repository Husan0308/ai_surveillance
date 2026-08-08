from .schemas import CameraPosition

class PositionResolver:
    def resolve(self,camera_id,frame_id,bbox,frame_width,frame_height,timestamp,global_id=None,local_track_id=None,pose_point=None):
        if frame_width<=0 or frame_height<=0 or bbox is None or len(bbox)<4:return None
        x,y=(float(bbox[0])+float(bbox[2]))*.5,float(bbox[3])
        x_norm=min(1,max(0,x/frame_width));y_norm=min(1,max(0,y/frame_height))
        key=str(global_id or f"{camera_id}:{local_track_id}")
        return CameraPosition(camera_id,frame_id,key,x_norm,y_norm,frame_width,frame_height,timestamp)
