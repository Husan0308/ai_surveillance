import inspect,unittest
from services.frontend import ui

class FrontendLayoutTests(unittest.TestCase):
 def test_camera_grid_is_always_two_columns(self):
  self.assertEqual([ui.camera_grid_position(i) for i in range(6)],[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)])
  self.assertEqual(ui.camera_grid_position(6),(3,0))
 def test_aspect_fit_and_letterboxed_overlay_mapping(self):
  for width,height in ((3200,1800),(2560,1440),(640,360)):
   rect=ui.aspect_fit_rect(1000,700,width,height);self.assertAlmostEqual(rect.width(),1000);self.assertAlmostEqual(rect.height(),562.5);self.assertAlmostEqual(rect.y(),68.75)
   box=ui.QRectF(width*.25,height*.25,width*.5,height*.5);mapped=ui.map_bbox_to_video_rect(rect,width,height,box)
   self.assertAlmostEqual(mapped.x(),250);self.assertAlmostEqual(mapped.y(),209.375);self.assertAlmostEqual(mapped.width(),500);self.assertAlmostEqual(mapped.height(),281.25)
 def test_settings_navigation_has_no_admin_gate(self):
  source=inspect.getsource(ui.MainWindow.navigate)
  for forbidden in ("PasswordDialog","unlocked","password","SURVEILLANCE_UI_ADMIN_PASSWORD"):self.assertNotIn(forbidden,source)
  self.assertFalse(hasattr(ui,"PasswordDialog"))

if __name__=="__main__":unittest.main()
