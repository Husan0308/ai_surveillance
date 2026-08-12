import os,sys,time,unittest
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from types import SimpleNamespace
import shiboken6
from PySide6.QtCore import QObject,Signal,QEvent,QCoreApplication
from PySide6.QtWidgets import QApplication
from services.frontend import ui
from services.frontend.async_api import AsyncApi

class Emitter(QObject):
    heatmap_updated=Signal(str)

class NoopAsync:
    def submit(self,*args,**kwargs):return None

class QtLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def hub(self):
        system=Emitter();system.async_api=NoopAsync();system.api=SimpleNamespace(get_heatmap=lambda *_:{})
        return SimpleNamespace(sys=system,snapshot=lambda *_:None,open_fullscreen=lambda *_:None,toast=lambda *_:None)

    def flush_delete(self):
        QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);self.app.processEvents()

    def test_camera_cards_rebuild_without_stale_heatmap_or_timer_callbacks(self):
        errors=[];previous=sys.excepthook;sys.excepthook=lambda *item:errors.append(item)
        try:
            hub=self.hub();sim=ui.CameraState("CAM-01","Camera 1","Room");sim.heat_on=True
            for _ in range(20):
                card=ui.CameraCard(sim,hub);hub.sys.heatmap_updated.emit("CAM-01");self.app.processEvents()
                card.dispose();self.assertFalse(card.rec_t.isActive());self.assertFalse(card.dot.t.isActive());self.assertNotIn(card.surface,sim.surfaces)
                card.deleteLater();self.flush_delete();self.assertFalse(shiboken6.isValid(card));hub.sys.heatmap_updated.emit("CAM-01");self.app.processEvents()
            self.assertEqual(errors,[])
        finally:sys.excepthook=previous

    def test_fullscreen_reopen_disposes_every_surface(self):
        hub=self.hub();sim=ui.CameraState("CAM-06","Camera 6","Room")
        for _ in range(20):
            dialog=ui.FullscreenCam(sim,hub);surface=dialog.surface;self.assertIn(surface,sim.surfaces)
            dialog.done(0);self.assertNotIn(surface,sim.surfaces);dialog.deleteLater();self.flush_delete();self.assertFalse(shiboken6.isValid(dialog))
        self.assertEqual(sim.surfaces,[])

    def test_async_owner_deletion_suppresses_callback(self):
        api=AsyncApi(SimpleNamespace());owner=QObject();called=[]
        api.submit(lambda:(time.sleep(.05),42)[1],lambda value:called.append(value),owner=owner)
        owner.deleteLater();self.flush_delete();api.pool.waitForDone(1000);self.app.processEvents();self.assertEqual(called,[]);api.shutdown()

if __name__=="__main__":unittest.main()
