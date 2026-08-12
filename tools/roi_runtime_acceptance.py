"""Exercise the real ROI canvas against CAM-05 MJPEG and restore its config."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QPoint,Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from roi_setup import CameraCanvas,FreezeFrameROIDialog
from services.frontend.api_client import ApiClient
from services.frontend.video_transport import MJPEGClient
from shared.settings import ServiceSettings

def main():
    app=QApplication.instance() or QApplication([]);api=ApiClient();camera=next(x for x in api.get_cameras() if x["id"]=="CAM-05");original=list(camera.get("recovery_rois",()))
    canvas=CameraCanvas(camera,lambda _canvas:None);canvas.resize(800,500);canvas.show();frames=[]
    client=MJPEGClient("CAM-05",ServiceSettings.from_env().ml_url.rstrip("/")+"/video/CAM-05")
    client.frame.connect(lambda _cid,fid,_stamp,image:(frames.append(fid),setattr(canvas,"frame",image),canvas.update()));client.start();QTest.qWait(5000)
    if len(set(frames))<3:raise RuntimeError(f"moving preview unavailable: {frames[-5:]}")
    frozen_frame=canvas.frame.copy();frozen_frame_id=frames[-1];dialog=FreezeFrameROIDialog(camera,frozen_frame);dialog.show();QTest.qWait(250);rect=dialog.canvas.image_rect()
    clicks=[QPoint(int(rect.left()+rect.width()*x),int(rect.top()+rect.height()*y)) for x,y in ((.2,.2),(.8,.2),(.8,.8),(.2,.8))]
    for point in clicks:QTest.mouseClick(dialog.canvas,Qt.LeftButton,Qt.NoModifier,point);QTest.qWait(50)
    dialog._undo();QTest.mouseClick(dialog.canvas,Qt.LeftButton,Qt.NoModifier,clicks[-1]);QTest.mouseMove(dialog.canvas,QPoint(int(rect.center().x()),int(rect.bottom()-10)));QTest.qWait(500)
    live_advanced_while_frozen=frames[-1]>frozen_frame_id;image=QImage(dialog.canvas.size(),QImage.Format_RGB32);image.fill(Qt.black);dialog.canvas.render(image);image.save("/tmp/roi_runtime_acceptance.png")
    polygon=dialog.polygon();mouse_events=dict(dialog.canvas.mouse_events);hover_visible=dialog.canvas.hover_point is not None;dialog.accept();payload=[*original,{"id":"runtime_acceptance","enabled":True,"polygon":polygon}]
    try:
        saved=api.update_camera("CAM-05",{"recovery_rois":payload});reloaded=next(x for x in api.get_cameras() if x["id"]=="CAM-05");match=next(x for x in reloaded["recovery_rois"] if x["id"]=="runtime_acceptance")
        print(json.dumps({"frames_received":len(frames),"first_frame":frames[0],"last_frame":frames[-1],"frozen_frame_id":frozen_frame_id,"live_advanced_while_frozen":live_advanced_while_frozen,"receive_fps":client.receive_fps,"points":len(polygon),"mouse_events":mouse_events,"hover_visible":hover_visible,"saved_points":len(match["polygon"]),"screenshot":"/tmp/roi_runtime_acceptance.png"},sort_keys=True))
    finally:
        api.update_camera("CAM-05",{"recovery_rois":original});client.stop();canvas.close()
    return 0

if __name__=="__main__":raise SystemExit(main())
