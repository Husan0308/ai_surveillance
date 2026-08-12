import inspect,unittest
from services.frontend import ui
from services.frontend.video_transport import LatestDecodedFrame

class FrontendLayoutTests(unittest.TestCase):
 def test_camera_grid_is_always_two_columns(self):
  self.assertEqual([ui.camera_grid_position(i) for i in range(6)],[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)])
  self.assertEqual(ui.camera_grid_position(6),(3,0))
 def test_aspect_fit_and_letterboxed_overlay_mapping(self):
  for width,height in ((3200,1800),(2560,1440),(640,360)):
   rect=ui.aspect_fit_rect(1000,700,width,height);self.assertAlmostEqual(rect.width(),1000);self.assertAlmostEqual(rect.height(),562.5);self.assertAlmostEqual(rect.y(),68.75)
   box=ui.QRectF(width*.25,height*.25,width*.5,height*.5);mapped=ui.map_bbox_to_video_rect(rect,width,height,box)
   self.assertAlmostEqual(mapped.x(),250);self.assertAlmostEqual(mapped.y(),209.375);self.assertAlmostEqual(mapped.width(),500);self.assertAlmostEqual(mapped.height(),281.25)
 def test_bbox_mapping_clamps_edges_for_all_source_sizes_and_cards(self):
  cases=((2560,1440),(3200,1800),(1280,720),(640,360))
  cards=((1000,700),(420,700),(1600,900),(777,333))
  for sw,sh in cases:
   for cw,ch in cards:
    video=ui.aspect_fit_rect(cw,ch,1280,720)
    for box in (ui.QRectF(-50,100,200,300),ui.QRectF(sw-100,100,300,300),ui.QRectF(100,-50,300,200),ui.QRectF(100,sh-100,300,300)):
     mapped=ui.map_bbox_to_video_rect(video,sw,sh,box)
     self.assertGreaterEqual(mapped.left(),video.left()-1e-6);self.assertGreaterEqual(mapped.top(),video.top()-1e-6)
     self.assertLessEqual(mapped.right(),video.right()+1e-6);self.assertLessEqual(mapped.bottom(),video.bottom()+1e-6)

 def test_letterbox_transform_uses_one_uniform_scale(self):
  video=ui.QRectF(0,0,1000,700);box=ui.QRectF(0,0,3200,1800)
  mapped=ui.map_bbox_to_video_rect(video,3200,1800,box)
  self.assertAlmostEqual(mapped.width(),1000);self.assertAlmostEqual(mapped.height(),562.5);self.assertAlmostEqual(mapped.top(),68.75)

 def test_label_rect_is_inside_rendered_image(self):
  image=ui.QRectF(20,30,400,225)
  for box in (ui.QRectF(20,30,50,100),ui.QRectF(390,100,30,100),ui.QRectF(100,220,50,35)):
   label=ui.clamped_label_rect(box,image,140,24)
   self.assertTrue(image.contains(label))

 def test_active_people_count_deduplicates_global_identity(self):
  from types import SimpleNamespace
  same=[SimpleNamespace(global_id="UNK 1",track_id="T1",known=False)]
  cameras=[SimpleNamespace(id="CAM-01",online=True,tracks=same),SimpleNamespace(id="CAM-02",online=True,tracks=[SimpleNamespace(global_id="UNK 1",track_id="T9",known=False)])]
  self.assertEqual(ui.active_global_counts(cameras),(0,1))

 def test_global_unknown_display_name_remains_unknown(self):
  track=ui.RealtimeTrack({"local_track_id":"T9","global_id":"UNK 2","display_name":"UNK 2","bbox":[0,0,10,20]})
  self.assertEqual(track.name,"UNK 2");self.assertFalse(track.known)

 def test_unknown_label_is_compact(self):
  self.assertEqual(ui.compact_unknown_label("UNK: TRACK-00004"),"UNK 00004")
  self.assertLessEqual(len(ui.compact_unknown_label("UNKNOWN-very-long-internal-id")),12)

 def test_metadata_dedupes_local_tracks_and_keeps_source_dimensions(self):
  state=ui.CameraState("CAM-01","Camera 1","");state.frame_id=10
  message={"frame_id":9,"timestamp":10.0,"frame_width":3200,"frame_height":1800,"tracks":[
   {"local_track_id":"T1","bbox":[0,0,10,10],"confidence":.5},
   {"local_track_id":"T1","bbox":[1,1,20,20],"confidence":.8}]}
  self.assertTrue(state.set_metadata(message));self.assertEqual(len(state.tracks),1)
  self.assertEqual((state.metadata_frame_width,state.metadata_frame_height),(3200,1800));self.assertEqual(state.tracks[0].conf,.8)

 def test_delayed_provisional_metadata_is_canonicalized_after_alias(self):
  state=ui.CameraState("CAM-04","Camera 4","");state.frame_id=20;aliases={"UNK 37":"UNK 5"};state.canonicalize_global_id=lambda value:aliases.get(value,value)
  message={"frame_id":19,"identity_version":1,"timestamp":10.0,"tracks":[{"local_track_id":"T9","global_id":"UNK 37","display_name":"UNK 37","bbox":[0,0,10,20]}]}
  self.assertTrue(state.set_metadata(message));self.assertEqual(state.tracks[0].global_id,"UNK 5");self.assertEqual(state.tracks[0].name,"UNK 5");self.assertEqual(state.identity_version,1)

 def test_fullscreen_accepts_ai_metadata_from_independent_frame_ids(self):
  state=ui.CameraState("CAM-06","Camera 6","");state.frame_id=3;state.independent_display_frame_domain=True
  message={"frame_id":900,"timestamp":10.0,"frame_width":640,"frame_height":360,"tracks":[{"local_track_id":"T1","global_id":"UNK 3","bbox":[64,36,320,180]}]}
  self.assertTrue(state.set_metadata(message));self.assertEqual(state.tracks[0].global_id,"UNK 3");self.assertEqual((state.metadata_frame_width,state.metadata_frame_height),(640,360))
  for width,height in ((640,360),(1280,720),(2560,1440),(3200,1800)):
   video=ui.aspect_fit_rect(1600,900,width,height);mapped=ui.map_bbox_to_video_rect(video,640,360,ui.QRectF(64,36,256,144));self.assertAlmostEqual(mapped.x(),160);self.assertAlmostEqual(mapped.y(),90);self.assertAlmostEqual(mapped.width(),640);self.assertAlmostEqual(mapped.height(),360)

 def test_identity_epoch_rollover_clears_stale_aliases(self):
  system=ui.System.__new__(ui.System);system.identity_aliases={"UNK 2":"UNK 1"};system.identity_version=7;system.identity_runtime_epoch="old";system.metadata_buffer=__import__("services.frontend.video_renderer",fromlist=["MetadataBuffer"]).MetadataBuffer();state=ui.CameraState("CAM-01","Camera 1","");state.identity_version=7;system.sims=[state]
  system._accept_identity_epoch({"identity_runtime_epoch":"new"});self.assertEqual(system.identity_aliases,{});self.assertEqual(system.identity_version,0);self.assertEqual(state.identity_version,0)

 def test_frontend_does_not_expire_tracks_before_backend(self):
  import time
  state=ui.CameraState("CAM-01","Camera 1","");state.frame_id=10
  state.set_metadata({"frame_id":9,"timestamp":time.time()-30,"tracks":[{"local_track_id":"T1","global_id":"UNK 1","bbox":[0,0,20,40],"observation_type":"predicted"}]})
  self.assertEqual(len(state.people),1)
  state.set_metadata({"frame_id":10,"timestamp":time.time(),"tracks":[]});self.assertEqual(state.people,[])

 def test_tracker_prediction_is_not_double_smoothed_in_ui(self):
  previous=ui.RealtimeTrack({"local_track_id":"T1","bbox":[0,0,20,40]},metadata_timestamp=1.0)
  predicted=ui.RealtimeTrack({"local_track_id":"T1","bbox":[10,0,30,40],"observation_type":"predicted","prediction_age_ms":200},previous,1.2)
  box=predicted.bbox(640,360,1.3);self.assertEqual((box.x(),box.width()),(10.0,20.0));self.assertEqual(predicted.observation_type,"predicted")

 def test_bbox_projects_to_display_frame_timestamp_and_honors_horizon(self):
  track=ui.RealtimeTrack({"local_track_id":"T1","bbox":[0,0,20,40],"velocity":[100,0,0,0],"state_timestamp":10.0,"last_detection_timestamp":10.0,"visual_expires_at":10.8},metadata_timestamp=10.0)
  box=track.bbox(640,360,10.5);self.assertAlmostEqual(box.x(),50.0);self.assertEqual(track.last_visual_time_error_ms,0.0);self.assertAlmostEqual(track.last_visual_age_before_ms,500.0)
  self.assertTrue(track.visible_at(10.8));self.assertFalse(track.visible_at(10.81))
  earlier=track.bbox(640,360,9.8);self.assertAlmostEqual(earlier.x(),0.0);self.assertEqual(track.last_visual_time_error_ms,0.0)

 def test_authorized_expiry_removes_visual_without_new_metadata(self):
  state=ui.CameraState("CAM-01","Camera 1","");state.frame_id=10;state.frame_timestamp=10.0
  state.set_metadata({"frame_id":10,"timestamp":10.0,"tracks":[{"local_track_id":"T1","bbox":[0,0,20,40],"state_timestamp":10.0,"visual_expires_at":10.5}]});self.assertEqual(len(state.people),1)
  state.frame_timestamp=10.51;self.assertEqual(state.people,[])

 def test_out_of_order_metadata_cannot_resurrect_expired_track(self):
  state=ui.CameraState("CAM-01","Camera 1","");state.frame_id=20
  self.assertTrue(state.set_metadata({"frame_id":18,"timestamp":18.0,"tracks":[{"local_track_id":"T1","bbox":[0,0,20,40]}]}))
  self.assertTrue(state.set_metadata({"frame_id":20,"timestamp":20.0,"tracks":[]}))
  self.assertFalse(state.set_metadata({"frame_id":19,"timestamp":19.0,"tracks":[{"local_track_id":"T1","bbox":[0,0,20,40]}]}));self.assertEqual(state.people,[])

 def test_final_overlay_is_unique_per_track_and_detection_wins(self):
  payloads=ui.unique_overlay_payloads(({"local_track_id":"T1","bbox":[1,1,9,9],"observation_type":"predicted"},{"local_track_id":"T1","bbox":[2,2,10,10],"observation_type":"detected"},{"local_track_id":"T2","bbox":[20,2,30,10],"observation_type":"detected"}))
  self.assertEqual([item["local_track_id"] for item in payloads],["T1","T2"]);self.assertEqual(payloads[0]["observation_type"],"detected")
 def test_frontend_latest_frame_coalescing_never_builds_backlog(self):
  pending=LatestDecodedFrame();pending.put(("CAM-01",1));pending.put(("CAM-01",2));pending.put(("CAM-01",3))
  self.assertEqual(pending.replaced,2);self.assertEqual(pending.take(),("CAM-01",3));self.assertIsNone(pending.take())

 def test_deleted_qt_surface_is_pruned_before_update(self):
  from PySide6.QtCore import QCoreApplication,QEvent
  from PySide6.QtWidgets import QApplication,QWidget
  app=QApplication.instance() or QApplication([]);state=ui.CameraState("CAM-01","Camera 1","");deleted=QWidget();live=QWidget();state.surfaces=[deleted,live]
  deleted.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents();state.update_surfaces();self.assertEqual(state.surfaces,[live])

 def test_event_renderer_does_not_shadow_datetime(self):
  source=inspect.getsource(ui.RightPanel.add_event);self.assertNotIn("from datetime import datetime",source);self.assertIn("e.setdefault(\"cam\"",source);self.assertIn("e.setdefault(\"person\"",source)

 def test_settings_navigation_has_no_admin_gate(self):
  source=inspect.getsource(ui.MainWindow.navigate)
  for forbidden in ("PasswordDialog","unlocked","password","SURVEILLANCE_UI_ADMIN_PASSWORD"):self.assertNotIn(forbidden,source)
  self.assertFalse(hasattr(ui,"PasswordDialog"))

 def test_camera_local_visual_lifecycle_does_not_mutate_other_camera(self):
  left=ui.CameraState("CAM-01","One","");right=ui.CameraState("CAM-02","Two","");left.frame_id=right.frame_id=2
  payload=lambda frame:{"frame_id":frame,"timestamp":float(frame),"tracks":[{"local_track_id":"T1","track_generation":1,"global_id":"UNK 1","bbox":[0,0,20,40]}]}
  self.assertTrue(left.set_metadata(payload(1)));self.assertTrue(right.set_metadata(payload(1)))
  self.assertTrue(left.set_metadata({"frame_id":2,"timestamp":2.0,"tracks":[]}))
  self.assertEqual(left.people,[]);self.assertEqual(len(right.people),1);self.assertEqual(right.people[0].global_id,"UNK 1")

 def test_old_generation_cannot_remove_reused_local_track(self):
  state=ui.CameraState("CAM-01","One","");state.frame_id=4
  state.set_metadata({"frame_id":1,"timestamp":1.0,"tracks":[{"local_track_id":"T1","track_generation":1,"bbox":[0,0,20,40]}]})
  state.set_metadata({"frame_id":2,"timestamp":2.0,"tracks":[]})
  state.set_metadata({"frame_id":3,"timestamp":3.0,"tracks":[{"local_track_id":"T1","track_generation":2,"bbox":[10,0,30,40]}]})
  state.set_metadata({"frame_id":4,"timestamp":4.0,"tracks":[{"local_track_id":"T1","track_generation":1,"bbox":[0,0,20,40]}]})
  self.assertEqual(len(state.people),1);self.assertEqual(state.people[0].track_generation,2);self.assertEqual(state.generation_mismatch_dropped_total,1)

 def test_backend_boundary_hide_is_visual_only(self):
  state=ui.CameraState("CAM-01","One","");state.frame_id=1;state.frame_timestamp=1.0
  state.set_metadata({"frame_id":1,"timestamp":1.0,"tracks":[{"local_track_id":"T1","track_generation":1,"bbox":[0,0,20,40],"visual_visible":False,"boundary_exit":True,"visual_expires_at":2.0}]})
  self.assertEqual(state.people,[]);self.assertEqual(len(state.tracks),1);self.assertTrue(state.tracks[0].boundary_exit)

 def test_metadata_waits_for_video_tick_instead_of_repainting_independently(self):
  from services.frontend.video_renderer import MetadataBuffer
  from types import SimpleNamespace
  calls=[];camera=SimpleNamespace(id="CAM-01",set_metadata=lambda message:calls.append("metadata"),update_surfaces=lambda:calls.append("paint"))
  system=ui.System.__new__(ui.System);system.metadata_buffer=MetadataBuffer();system.sims=[camera];system.identity_runtime_epoch=None;system.identity_aliases={};system.identity_version=0
  message={"type":"frame.metadata","camera_id":"CAM-01","frame_id":7,"timestamp":7.0,"tracks":[]}
  system._on_remote_message(message)
  self.assertEqual(calls,[]);self.assertEqual(system.metadata_buffer.match("CAM-01",7,7.0)["frame_id"],7)

 def test_presented_frame_does_not_emit_redundant_online_repaint(self):
  from services.frontend.video_transport import MJPEGClient
  source=inspect.getsource(MJPEGClient._emit_latest)
  self.assertNotIn("online.emit(camera_id,True)",source)
  self.assertIn("self.frame.emit(camera_id,frame_id,timestamp,image)",source)

 def test_repeated_offline_status_is_edge_triggered(self):
  from types import SimpleNamespace
  calls=[];camera=SimpleNamespace(id="CAM-01",online=False,frame=None,clear_frame=lambda:calls.append("clear"),update_surfaces=lambda:calls.append("paint"))
  system=ui.System.__new__(ui.System);system.sims=[camera]
  system._on_video_status("CAM-01",False);self.assertEqual(calls,[])
  camera.online=True;camera.frame=object();system._on_video_status("CAM-01",False);self.assertEqual(calls,["clear","paint"])

 def test_camera_card_style_rebuild_is_state_transition_only(self):
  source=inspect.getsource(ui.CameraCard.refresh)
  self.assertIn("offline!=self._last_offline",source)
  self.assertNotIn('self.setProperty("offline", not on)',source)

if __name__=="__main__":unittest.main()
