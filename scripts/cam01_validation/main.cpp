// CAM-01-only diagnostic executable. No pixel mapping, inference or Python callbacks.
#include <gst/gst.h>
#include <gst/video/videooverlay.h>
#include <glib-unix.h>
#include <X11/Xlib.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sys/resource.h>
#include <unistd.h>

struct Counter {
  std::atomic<guint64> frames{0}, pts{GST_CLOCK_TIME_NONE}, backwards{0}, duplicates{0};
  std::atomic<gint64> last_us{0}, max_gap_us{0};
};
static Counter decoded, delivered;
static GstElement *pipeline, *source, *queue_element, *sink_element;
static GMainLoop *loop;
static Display *display;
static Window window;
static std::string uri, username, password;
static bool decode_only=false, record_mode=false, stopping=false;
static std::atomic_bool fatal{false};
static guint errors=0, warnings=0, recoveries=0, pending_restart=0;
static gint64 start_us, previous_us, last_message_us=0, last_restart_us=0;
static guint64 previous_frames=0;
static double previous_cpu=0;
static int duration_sec=30, latency_ms=300;
static std::atomic_bool decoder_seen{false};
static GstElement *jitterbuffer=nullptr;
static GMutex jitter_lock;

static void event(const char *kind, const std::string &detail) {
  std::string safe=detail;
  for (const auto &secret : {password, uri}) {
    if (secret.empty()) continue;
    size_t at=0;
    while ((at=safe.find(secret,at)) != std::string::npos) {
      safe.replace(at,secret.size(),"<redacted>"); at+=10;
    }
  }
  for (auto &c:safe) if (c=='\n' || c=='\r') c=' ';
  g_print("CAM-01 %s %s\n",kind,safe.c_str());
}
static void fail(const std::string &detail) {
  event("FATAL", detail); fatal=true; if(loop) g_main_loop_quit(loop);
}
static GstElement *make(const char *factory, const char *name) {
  GstElement *e=gst_element_factory_make(factory,name);
  if(!e) { event("FATAL",std::string("missing plugin ")+factory); std::exit(2); }
  gst_bin_add(GST_BIN(pipeline),e); return e;
}
static void link(GstElement *a,GstElement *b) {
  if(!gst_element_link(a,b)) { event("FATAL","static link failed"); std::exit(2); }
}
static GstPadProbeReturn count_buffer(GstPad *pad,GstPadProbeInfo *info,gpointer data) {
  auto *c=static_cast<Counter*>(data);
  GstBuffer *b=GST_PAD_PROBE_INFO_BUFFER(info);
  if(!b) return GST_PAD_PROBE_OK;
  gint64 now=g_get_monotonic_time(), last=c->last_us.exchange(now);
  if(last && now-last>c->max_gap_us.load()) c->max_gap_us=now-last;
  guint64 pts=GST_BUFFER_PTS(b), before=c->pts.exchange(pts);
  if(pts!=GST_CLOCK_TIME_NONE && before!=GST_CLOCK_TIME_NONE) {
    if(pts<before) ++c->backwards;
    if(pts==before) ++c->duplicates;
  }
  if(c->frames.fetch_add(1)==0) {
    GstCaps *caps=gst_pad_get_current_caps(pad);
    gchar *s=caps ? gst_caps_to_string(caps) : g_strdup("unknown");
    event(c==&decoded ? "DECODE_CAPS" : "OUTPUT_CAPS",s);
    if(c==&decoded && (!caps || !gst_caps_features_contains(gst_caps_get_features(caps,0),"memory:NVMM")))
      fail("decoded video is not NVMM");
    g_free(s); if(caps) gst_caps_unref(caps);
  }
  return GST_PAD_PROBE_OK;
}
static void probe(GstElement *e,const char *pad_name,Counter *c) {
  GstPad *pad=gst_element_get_static_pad(e,pad_name);
  gst_pad_add_probe(pad,GST_PAD_PROBE_TYPE_BUFFER,count_buffer,c,nullptr);
  gst_object_unref(pad);
}
static void configure_rtsp(GstElement *e) {
  g_object_set(e,"user-id",username.c_str(),"user-pw",password.c_str(),
    "protocols",4,"latency",latency_ms,"drop-on-latency",FALSE,
    "tcp-timeout",guint64(10000000),"do-rtsp-keep-alive",TRUE,nullptr);
  // Preserve normal RTP/RTCP timestamps; do not force receive-time timestamps.
}
static void child_added(GstBin*,GstBin*,GstElement *e,gpointer) {
  GstElementFactory *f=gst_element_get_factory(e);
  if(!f) return;
  const char *name=gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(f));
  if(!std::strcmp(name,"rtspsrc")) {configure_rtsp(e); event("RTSP_CHILD","tcp=1 timeout_sec=10");}
  if(!std::strcmp(name,"nvv4l2decoder")) {
    decoder_seen=true; probe(e,"src",&decoded);
    g_object_set(e,"drop-frame-interval",0u,"skip-frames",0,nullptr);
    event("DECODER","nvv4l2decoder hardware=1 skip=0 drop=0");
  }
  if(!std::strcmp(name,"rtpjitterbuffer")) {
    g_mutex_lock(&jitter_lock);
    if(jitterbuffer) gst_object_unref(jitterbuffer);
    jitterbuffer=GST_ELEMENT(gst_object_ref(e));
    g_mutex_unlock(&jitter_lock);
  }
  if(!std::strcmp(name,"queue"))
    g_object_set(e,"leaky",0,"max-size-buffers",12u,"max-size-bytes",0u,"max-size-time",guint64(0),nullptr);
}
static void dynamic_pad(GstElement*,GstPad *pad,gpointer target) {
  GstCaps *caps=gst_pad_get_current_caps(pad);
  if(!caps) caps=gst_pad_query_caps(pad,nullptr);
  if(!caps || gst_caps_is_any(caps) || gst_caps_is_empty(caps)) {
    if(caps) gst_caps_unref(caps);
    fail("source pad has no usable video caps"); return;
  }
  auto *s=gst_caps_get_structure(caps,0);
  const char *media=gst_structure_get_name(s);
  bool is_video=decode_only ? (gst_structure_has_field(s,"media") &&
      !g_strcmp0(gst_structure_get_string(s,"media"),"video")) : g_str_has_prefix(media,"video/");
  if(is_video) {
    GstPad *dest=gst_element_get_static_pad(GST_ELEMENT(target),"sink");
    if(!gst_pad_is_linked(dest)) {
      if(gst_pad_link(pad,dest)!=GST_PAD_LINK_OK) fail("dynamic video link failed");
      else event("SOURCE_LINK",std::string(GST_PAD_NAME(pad))+" linked");
    }
    gst_object_unref(dest);
  }
  gst_caps_unref(caps);
}
static gboolean restart(gpointer) {
  pending_restart=0;
  if(stopping) return G_SOURCE_REMOVE;
  ++recoveries; last_restart_us=g_get_monotonic_time();
  event("RECOVERY","restarting single-camera graph in same process");
  gst_element_set_state(pipeline,GST_STATE_NULL);
  if(gst_element_set_state(pipeline,GST_STATE_PLAYING)==GST_STATE_CHANGE_FAILURE) fail("recovery PLAYING failed");
  return G_SOURCE_REMOVE;
}
static gboolean bus_message(GstBus*,GstMessage *m,gpointer) {
  if(GST_MESSAGE_TYPE(m)==GST_MESSAGE_ERROR || GST_MESSAGE_TYPE(m)==GST_MESSAGE_WARNING) {
    bool error=GST_MESSAGE_TYPE(m)==GST_MESSAGE_ERROR;
    GError *e=nullptr; gchar *debug=nullptr;
    if(error) {++errors; gst_message_parse_error(m,&e,&debug);} else {++warnings; gst_message_parse_warning(m,&e,&debug);}
    std::string message=std::string(GST_OBJECT_NAME(m->src))+": "+e->message;
    if(debug) message+=" | "+std::string(debug);
    gint64 now=g_get_monotonic_time();
    if(now-last_message_us>2000000 || errors+warnings<3) {event(error?"ERROR":"WARNING",message);last_message_us=now;}
    bool source_error=m->src==GST_OBJECT(source) || gst_object_has_as_ancestor(m->src,GST_OBJECT(source));
    if(error) {
      if(message.find("Unauthorized")!=std::string::npos || message.find("401")!=std::string::npos) fail("authentication rejected");
      else if(!source_error) fail("shared decode/mux/output error: "+message);
      else if(!pending_restart) pending_restart=g_timeout_add_seconds(5,restart,nullptr);
    }
    g_error_free(e); g_free(debug);
  } else if(GST_MESSAGE_TYPE(m)==GST_MESSAGE_EOS && stopping) {
    g_main_loop_quit(loop);
  } else if(GST_MESSAGE_TYPE(m)==GST_MESSAGE_EOS && !stopping) {
    event("EOS","recovering live source");
    if(!pending_restart) pending_restart=g_timeout_add_seconds(5,restart,nullptr);
  }
  return G_SOURCE_CONTINUE;
}
static double cpu_seconds() {
  rusage r{}; getrusage(RUSAGE_SELF,&r);
  return r.ru_utime.tv_sec+r.ru_utime.tv_usec/1e6+r.ru_stime.tv_sec+r.ru_stime.tv_usec/1e6;
}
static gboolean tick(gpointer) {
  gint64 now=g_get_monotonic_time();
  double elapsed=(now-start_us)/1e6, dt=(now-previous_us)/1e6, cpu=cpu_seconds();
  guint64 frames=delivered.frames.load(); guint q=0; guint64 qtime=0;
  g_object_get(queue_element,"current-level-buffers",&q,"current-level-time",&qtime,nullptr);
  long pages=0,resident=0; std::ifstream statm("/proc/self/statm"); statm>>pages>>resident;
  GstStructure *stats=nullptr; guint64 rendered=0,dropped=0,lost=0,late=0,pushed=0;
  g_object_get(sink_element,"stats",&stats,nullptr);
  if(stats) {gst_structure_get_uint64(stats,"rendered",&rendered);gst_structure_get_uint64(stats,"dropped",&dropped);gst_structure_free(stats);}
  g_mutex_lock(&jitter_lock);
  if(jitterbuffer) {
    g_object_get(jitterbuffer,"stats",&stats,nullptr);
    if(stats) {gst_structure_get_uint64(stats,"num-lost",&lost);gst_structure_get_uint64(stats,"num-late",&late);gst_structure_get_uint64(stats,"num-pushed",&pushed);gst_structure_free(stats);}
  }
  g_mutex_unlock(&jitter_lock);
  double age=delivered.last_us ? (now-delivered.last_us)/1e6 : elapsed;
  g_print("CAM-01 STATS elapsed=%.3f decoded=%lu output=%lu fps=%.3f age=%.3f pts_ns=%lu pts_backwards=%lu pts_duplicates=%lu max_gap_ms=%.3f queue=%u queue_ms=%.3f rss_mib=%.3f cpu_pct=%.3f rendered=%lu dropped=%lu rtp_lost=%lu rtp_late=%lu rtp_pushed=%lu errors=%u warnings=%u recoveries=%u\n",
    elapsed,decoded.frames.load(),frames,(frames-previous_frames)/dt,age,delivered.pts.load(),delivered.backwards.load(),delivered.duplicates.load(),delivered.max_gap_us/1000.0,q,qtime/1e6,resident*sysconf(_SC_PAGESIZE)/1048576.0,100*(cpu-previous_cpu)/dt,rendered,dropped,lost,late,pushed,errors,warnings,recoveries);
  previous_frames=frames; previous_us=now; previous_cpu=cpu;
  if(display) {while(XPending(display)) {XEvent e;XNextEvent(display,&e);if(e.type==DestroyNotify) {fail("display window closed");return G_SOURCE_REMOVE;}}}
  if(age>90) {fail("no advancing frames for 90 seconds");return G_SOURCE_REMOVE;}
  // Native nvurisrcbin owns ordinary reconnection. This bounds hard source/EOS failures.
  if(!decode_only && age>30 && !pending_restart && now-last_restart_us>30000000)
    pending_restart=g_timeout_add_seconds(5,restart,nullptr);
  if(elapsed>=duration_sec) {
    stopping=true;
    if(record_mode) {
      // nvurisrcbin consumes source EOS to support live reconnection. Finalize
      // the diagnostic output directly, leaving the source recovery policy intact.
      GstElement *encoder=gst_bin_get_by_name(GST_BIN(pipeline),"NVENC");
      gst_element_send_event(encoder,gst_event_new_eos()); gst_object_unref(encoder);
      g_timeout_add_seconds(5,[](gpointer)->gboolean {fail("output EOS did not complete");return G_SOURCE_REMOVE;},nullptr);
    } else g_main_loop_quit(loop);
    return G_SOURCE_REMOVE;
  }
  return G_SOURCE_CONTINUE;
}
static gboolean stop(gpointer) {stopping=true;g_main_loop_quit(loop);return G_SOURCE_REMOVE;}
static std::string secret(GKeyFile *f,const char *key) {
  gchar *s=g_key_file_get_string(f,"camera",key,nullptr);
  if(!s) return "";
  gsize n=0;guchar *raw=g_base64_decode(s,&n);std::string v(reinterpret_cast<char*>(raw),n);g_free(raw);g_free(s);return v;
}
int main(int argc,char **argv) {
  if(argc!=4) {g_printerr("usage: cam01-validator CONFIG DURATION decode|display|record\n");return 2;}
  GKeyFile *f=g_key_file_new();GError *err=nullptr;
  if(!g_key_file_load_from_file(f,argv[1],G_KEY_FILE_NONE,&err)) {g_printerr("CAM-01 config read failed\n");return 2;}
  uri=secret(f,"uri");username=secret(f,"username");password=secret(f,"password");
  duration_sec=std::atoi(argv[2]);decode_only=!std::strcmp(argv[3],"decode");record_mode=!std::strcmp(argv[3],"record");
  latency_ms=g_key_file_get_integer(f,"camera","latency_ms",nullptr);g_key_file_free(f);
  if(uri.rfind("rtsp://",0)!=0 || duration_sec<5 || latency_ms<100) return 2;
  gst_init(nullptr,nullptr);g_mutex_init(&jitter_lock);loop=g_main_loop_new(nullptr,FALSE);
  pipeline=gst_pipeline_new("CAM-01-validation");
  queue_element=make("queue","CAM-01-queue");
  g_object_set(queue_element,"max-size-buffers",12u,"max-size-bytes",0u,"max-size-time",guint64(0),"leaky",0,nullptr);
  if(decode_only) {
    source=make("rtspsrc","CAM-01-rtsp");configure_rtsp(source);g_object_set(source,"location",uri.c_str(),nullptr);
    auto *depay=make("rtph264depay","depay"), *parse=make("h264parse","parse"), *dec=make("nvv4l2decoder","NVDEC");
    g_object_set(dec,"gpu-id",0u,"cudadec-memtype",0,"num-extra-surfaces",4u,"drop-frame-interval",0u,"skip-frames",0,nullptr);
    decoder_seen=true;probe(dec,"src",&decoded);event("DECODER","nvv4l2decoder explicit H264 NVDEC");
    g_signal_connect(source,"pad-added",G_CALLBACK(dynamic_pad),depay);link(depay,parse);link(parse,dec);link(dec,queue_element);
    sink_element=make("fakesink","decode-verification");link(queue_element,sink_element);
  } else {
    source=make("nvurisrcbin","CAM-01-source");
    g_signal_connect(source,"deep-element-added",G_CALLBACK(child_added),nullptr);
    g_signal_connect(source,"pad-added",G_CALLBACK(dynamic_pad),queue_element);
    g_object_set(source,"uri",uri.c_str(),"gpu-id",0u,"source-id",0,"disable-audio",TRUE,"select-rtp-protocol",4,
      "latency",guint(latency_ms),"drop-on-latency",FALSE,"num-extra-surfaces",4u,"cudadec-memtype",0,
      "drop-frame-interval",0u,"dec-skip-frames",0,"low-latency-mode",FALSE,
      "rtsp-reconnect-interval",5u,"init-rtsp-reconnect-interval",5u,"rtsp-reconnect-attempts",-1,
      "max-size-buffers",12u,"leaky",0,"message-forward",TRUE,nullptr);
    auto *mux=make("nvstreammux","mux");
    g_object_set(mux,"batch-size",1u,"width",2560u,"height",1440u,"live-source",TRUE,
      "batched-push-timeout",50000,"sync-inputs",FALSE,"attach-sys-ts",TRUE,
      "nvbuf-memory-type",2,"cache-buffer",FALSE,"drop-pipeline-eos",TRUE,nullptr);
    GstPad *mp=gst_element_request_pad_simple(mux,"sink_0"),*qp=gst_element_get_static_pad(queue_element,"src");
    if(!mp || gst_pad_link(qp,mp)!=GST_PAD_LINK_OK) return 2;
    gst_object_unref(qp);gst_object_unref(mp);
    auto *conv=make("nvvideoconvert","gpu-convert"),*caps=make("capsfilter","rgba-nvmm");
    GstCaps *video_caps=gst_caps_from_string(record_mode ? "video/x-raw(memory:NVMM),format=NV12" : "video/x-raw(memory:NVMM),format=RGBA");
    g_object_set(caps,"caps",video_caps,nullptr);gst_caps_unref(video_caps);
    link(mux,conv);link(conv,caps);
    if(record_mode) {
      auto *enc=make("nvv4l2h264enc","NVENC"), *parse=make("h264parse","output-h264"), *filemux=make("matroskamux","recording");
      g_object_set(enc,"bitrate",6000000u,"iframeinterval",20u,"idrinterval",20u,"insert-sps-pps",TRUE,nullptr);
      sink_element=make("filesink","CAM-01-recording");g_object_set(sink_element,"location","/work/CAM-01.mkv",nullptr);
      link(caps,enc);link(enc,parse);link(parse,filemux);link(filemux,sink_element);
      probe(parse,"src",&delivered);
      event("GRAPH","nvurisrcbin/NVDEC->NVMM->queue->nvstreammux(batch=1,2560x1440,live,50000us)->nvvideoconvert/NVMM->NVENC->h264parse->matroskamux->CAM-01.mkv inference=0");
    } else {
      sink_element=make("nveglglessink","CAM-01-display");link(caps,sink_element);
      XInitThreads();display=XOpenDisplay(nullptr);
      if(!display) {event("FATAL","X11 display unavailable");return 2;}
      window=XCreateSimpleWindow(display,DefaultRootWindow(display),40,40,1280,720,0,0,0);
      XStoreName(display,window,"CAM-01 | DeepStream 9.1 | camera-only");XSelectInput(display,window,StructureNotifyMask);
      XMapWindow(display,window);XSync(display,FALSE);
      gst_video_overlay_set_window_handle(GST_VIDEO_OVERLAY(sink_element),window);
      g_print("CAM-01 WINDOW id=%lu width=1280 height=720\n",window);
      event("GRAPH","nvurisrcbin/NVDEC->NVMM->queue->nvstreammux(batch=1,2560x1440,live,50000us)->nvvideoconvert->RGBA/NVMM->EGL inference=0");
    }
  }
  g_object_set(sink_element,"sync",FALSE,"async",FALSE,"qos",FALSE,"max-lateness",gint64(-1),"enable-last-sample",FALSE,nullptr);
  if(!record_mode) probe(sink_element,"sink",&delivered);
  GstBus *bus=gst_element_get_bus(pipeline);gst_bus_add_watch(bus,bus_message,nullptr);gst_object_unref(bus);
  g_unix_signal_add(SIGINT,stop,nullptr);g_unix_signal_add(SIGTERM,stop,nullptr);
  start_us=previous_us=last_restart_us=g_get_monotonic_time();previous_cpu=cpu_seconds();
  if(gst_element_set_state(pipeline,GST_STATE_PLAYING)==GST_STATE_CHANGE_FAILURE) fail("initial PLAYING failed");
  g_timeout_add_seconds(5,tick,nullptr);
  if(!fatal) g_main_loop_run(loop);
  stopping=true;gst_element_set_state(pipeline,GST_STATE_NULL);
  gst_object_unref(pipeline);if(jitterbuffer)gst_object_unref(jitterbuffer);
  if(display) {XDestroyWindow(display,window);XCloseDisplay(display);}
  g_main_loop_unref(loop);
  g_print("CAM-01 END output=%lu hardware_decoder=%d fatal=%d errors=%u warnings=%u\n",delivered.frames.load(),int(decoder_seen.load()),int(fatal.load()),errors,warnings);
  return fatal || !decoder_seen || delivered.frames==0 ? 1 : 0;
}
