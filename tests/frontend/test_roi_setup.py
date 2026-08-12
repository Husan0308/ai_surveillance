import os,unittest
from unittest.mock import patch
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from PySide6.QtCore import QPoint,Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import roi_setup

CAMERAS=[{"id":f"CAM-{index:02d}","enabled":True,"recovery_rois":[]} for index in range(1,7)]

class ROISetupTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
 def immediate(self,call,on_success=None,on_error=None,owner=None):
  try:value=call()
  except Exception as exc:
   if on_error:on_error(str(exc))
  else:
   if on_success:on_success(value)
 def test_startup_loads_six_cameras_without_opening_rtsp(self):
  with patch.object(roi_setup.ApiClient,"get_cameras",return_value=CAMERAS),patch.object(roi_setup.AsyncApi,"submit",self.immediate),patch.object(roi_setup.MJPEGClient,"start"),patch.object(roi_setup.MJPEGClient,"stop"):
   window=roi_setup.ROISetupWindow();self.assertEqual(list(window.cards),[f"CAM-{i:02d}" for i in range(1,7)]);self.assertEqual(len(window.clients),6);window.close()

 def test_mouse_first_point_preview_resize_mapping_and_repaint(self):
  canvas=roi_setup.CameraCanvas({"id":"CAM-05","recovery_rois":[{"id":"far","enabled":True,"polygon":[]}]},lambda _canvas:None);canvas.resize(800,500);canvas.frame=QImage(640,360,QImage.Format_RGB32);canvas.frame.fill(Qt.black);canvas.current=0;canvas.begin_drawing();canvas.show();self.app.processEvents()
  QTest.mouseClick(canvas,Qt.LeftButton,Qt.NoModifier,QPoint(400,250));self.app.processEvents();self.assertEqual(len(canvas.points),1);self.assertAlmostEqual(canvas.points[0].x(),.5,places=2);self.assertEqual(canvas.mouse_events["press"],1)
  QTest.mouseMove(canvas,QPoint(600,300));self.app.processEvents();self.assertIsNotNone(canvas.hover_point);self.assertGreater(canvas.mouse_events["move"],0)
  painted=QImage(canvas.size(),QImage.Format_RGB32);painted.fill(Qt.black);canvas.render(painted);color=painted.pixelColor(400,250);self.assertGreater(color.green(),100)
  normalized=canvas.points[0];canvas.resize(1000,400);self.app.processEvents();screen=canvas.screen_at(normalized);self.assertAlmostEqual(canvas.normalized_at(screen).x(),normalized.x(),places=5);self.assertAlmostEqual(canvas.normalized_at(screen).y(),normalized.y(),places=5);canvas.close()

 def test_freeze_frame_dialog_is_large_static_and_draws_exact_points(self):
  source=QImage(640,360,QImage.Format_RGB32);source.fill(Qt.black);dialog=roi_setup.FreezeFrameROIDialog({"id":"CAM-05","recovery_rois":[]},source);dialog.show();self.app.processEvents()
  self.assertGreaterEqual(dialog.canvas.minimumWidth(),900);source.fill(Qt.white);self.assertNotEqual(dialog.canvas.frame.pixelColor(0,0),source.pixelColor(0,0))
  rect=dialog.canvas.image_rect();positions=[rect.center().toPoint(),QPoint(int(rect.left()+rect.width()*.7),int(rect.top()+rect.height()*.4)),QPoint(int(rect.left()+rect.width()*.6),int(rect.top()+rect.height()*.7))]
  for point in positions:QTest.mouseClick(dialog.canvas,Qt.LeftButton,Qt.NoModifier,point);self.app.processEvents()
  self.assertEqual(len(dialog.canvas.points),3);self.assertTrue(dialog.done_button.isEnabled());self.assertEqual(dialog.points_label.text(),"Points: 3")
  first=dialog.canvas.points[0];screen=dialog.canvas.screen_at(first);self.assertAlmostEqual(screen.x(),positions[0].x(),delta=1);self.assertAlmostEqual(screen.y(),positions[0].y(),delta=1)
  dialog._undo();self.assertEqual(len(dialog.canvas.points),2);self.assertFalse(dialog.done_button.isEnabled());dialog.reject()


if __name__=="__main__":unittest.main()
