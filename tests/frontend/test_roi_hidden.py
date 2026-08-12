import inspect,unittest
from services.frontend.roi_editor import ROIFrameCanvas
from services.frontend.ui import VideoSurface,FullscreenCam

class ROIHiddenTests(unittest.TestCase):
    def test_roi_geometry_is_painted_only_by_settings_editor(self):
        self.assertIn("drawLine",inspect.getsource(ROIFrameCanvas.paintEvent))
        self.assertNotIn("recovery_rois",inspect.getsource(VideoSurface))
        self.assertNotIn("recovery_rois",inspect.getsource(FullscreenCam))

if __name__=="__main__":unittest.main()
