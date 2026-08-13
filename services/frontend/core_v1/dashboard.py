from __future__ import annotations

from collections import deque
import http.client,json,threading,time
from datetime import datetime
from PySide6.QtCore import Qt,QTimer,QSize
from PySide6.QtGui import QFont,QIcon,QImage,QPainter,QPen,QPixmap,QColor
from PySide6.QtWidgets import QApplication,QFrame,QGridLayout,QHBoxLayout,QHeaderView,QLabel,QMainWindow,QPushButton,QStackedWidget,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget

CAMERAS=[f'CAM-{i:02d}' for i in range(1,7)];ML_HOST='127.0.0.1';ML_PORT=8001
BG='#000a1c';SIDEBAR='#000f27';PANEL='#001126';CARD='#00162f';BORDER='#073154';TEXT='#f6f8fc';MUTED='#a8b5c8';BLUE='#0d63ff';GREEN='#00e676';ORANGE='#ff8a00';RED='#ff334e';CYAN='#16b9ff'
TITLES={'CAM-01':'Office 1 (A)','CAM-02':'Office 2 (A)','CAM-03':'Office 3 (A)','CAM-04':'Office 1 (B)','CAM-05':'Office 2 (B)','CAM-06':'Office 3 (B)'}

def font(s,w=QFont.Weight.Normal):f=QFont('Inter');f.setPixelSize(s);f.setWeight(w);return f

def simple_icon(kind,color=TEXT,size=22):
    pm=QPixmap(size,size);pm.fill(Qt.transparent);p=QPainter(pm);p.setRenderHint(QPainter.Antialiasing);p.setPen(QPen(QColor(color),2));s=size
    if kind=='menu':
        for y in (.3,.5,.7):p.drawLine(int(.18*s),int(y*s),int(.82*s),int(y*s))
    elif kind=='person':p.drawEllipse(int(.38*s),int(.12*s),int(.24*s),int(.24*s));p.drawArc(int(.2*s),int(.42*s),int(.6*s),int(.45*s),20*16,140*16)
    elif kind=='camera':p.drawRect(int(.15*s),int(.27*s),int(.55*s),int(.42*s));p.drawLine(int(.7*s),int(.39*s),int(.88*s),int(.29*s));p.drawLine(int(.88*s),int(.29*s),int(.88*s),int(.69*s));p.drawLine(int(.88*s),int(.69*s),int(.7*s),int(.58*s))
    elif kind=='bell':p.drawArc(int(.25*s),int(.15*s),int(.5*s),int(.55*s),0,180*16);p.drawLine(int(.25*s),int(.43*s),int(.25*s),int(.68*s));p.drawLine(int(.75*s),int(.43*s),int(.75*s),int(.68*s));p.drawLine(int(.2*s),int(.68*s),int(.8*s),int(.68*s))
    else:p.drawRect(int(.2*s),int(.2*s),int(.6*s),int(.6*s))
    p.end();return QIcon(pm)

class FrameReader:
    def __init__(self,cid):self.cid=cid;self.stop_flag=threading.Event();self.lock=threading.Lock();self.image=None;self.version=-1;self.frames=0;self.last=0.;self.thread=None
    def start(self):self.thread=threading.Thread(target=self.run,daemon=True,name=f'ui-frame-{self.cid}');self.thread.start()
    def stop(self):self.stop_flag.set()
    def latest(self):
        with self.lock:return self.image,self.version
    def run(self):
        conn=None;version=-1
        while not self.stop_flag.is_set():
            try:
                if conn is None:conn=http.client.HTTPConnection(ML_HOST,ML_PORT,timeout=2)
                conn.request('GET',f'/frame/{self.cid}?after={version}&wait_ms=180',headers={'Connection':'keep-alive','Cache-Control':'no-cache'});r=conn.getresponse();data=r.read()
                if r.status!=200:raise RuntimeError(r.status)
                nxt=int(r.getheader('X-Frame-Version') or version+1)
                if nxt<=version:continue
                img=QImage.fromData(data,'JPG')
                if img.isNull():continue
                version=nxt
                with self.lock:self.image=img;self.version=version;self.frames+=1;self.last=time.monotonic()
            except Exception:
                if conn:
                    try:conn.close()
                    except Exception:pass
                conn=None;self.stop_flag.wait(.25)

class LiveState:
    def __init__(self):self.stop_flag=threading.Event();self.lock=threading.Lock();self.state={'connected':False,'health':{},'detections':{},'reid':{}};self.recent=deque(maxlen=20);self.events=deque(maxlen=80);self.seen={}
    def start(self):threading.Thread(target=self.run,daemon=True,name='ui-state').start()
    def stop(self):self.stop_flag.set()
    def snapshot(self):
        with self.lock:return dict(self.state),list(self.recent),list(self.events)
    def get(self,conn,path):conn.request('GET',path,headers={'Connection':'keep-alive','Cache-Control':'no-cache'});r=conn.getresponse();b=r.read();
    def _json(self,conn,path):
        conn.request('GET',path,headers={'Connection':'keep-alive','Cache-Control':'no-cache'});r=conn.getresponse();b=r.read()
        if r.status!=200:raise RuntimeError(r.status)
        return json.loads(b.decode())
    def observe(self,reid):
        now=datetime.now();cams=((reid.get('state') or {}).get('cameras') or {})
        for cid,tracks in cams.items():
            for t in tracks or []:
                gid=str(t.get('global_id') or '')
                if not gid:continue
                key=(cid,int(t.get('local_id') or 0));old=self.seen.get(key);self.seen[key]=gid
                if old==gid:continue
                e={'time':now.strftime('%H:%M:%S'),'camera':cid,'gid':gid,'reason':str(t.get('reason') or 'detected'),'similarity':t.get('similarity')};self.recent.appendleft(e);self.events.appendleft(e)
    def run(self):
        conn=None
        while not self.stop_flag.is_set():
            try:
                if conn is None:conn=http.client.HTTPConnection(ML_HOST,ML_PORT,timeout=1.5)
                h=self._json(conn,'/health');d=self._json(conn,'/detections');r=self._json(conn,'/reid');self.observe(r)
                with self.lock:self.state={'connected':True,'health':h,'detections':d,'reid':r}
                self.stop_flag.wait(.35)
            except Exception:
                if conn:
                    try:conn.close()
                    except Exception:pass
                conn=None
                with self.lock:self.state={**self.state,'connected':False}
                self.stop_flag.wait(.7)

class CameraImage(QLabel):
    def __init__(self):super().__init__('Connecting...');self.img=None;self.setAlignment(Qt.AlignCenter);self.setStyleSheet('background:#000;color:#8192aa;border:0')
    def set_frame(self,img):self.img=img;self.apply()
    def apply(self):
        if self.img is not None and self.width()>2 and self.height()>2:self.setPixmap(QPixmap.fromImage(self.img).scaled(self.size(),Qt.KeepAspectRatio,Qt.FastTransformation))
    def resizeEvent(self,e):super().resizeEvent(e);self.apply()

class CameraTile(QFrame):
    def __init__(self,cid,n):
        super().__init__();self.setObjectName('cameraTile');o=QVBoxLayout(self);o.setContentsMargins(0,0,0,0);o.setSpacing(0);self.head=QWidget();h=QHBoxLayout(self.head);h.setContentsMargins(8,3,8,3);num=QLabel(f'{n:02d}');num.setStyleSheet(f'background:{BLUE};border-radius:5px;padding:4px');h.addWidget(num);h.addWidget(QLabel(TITLES[cid]));h.addStretch();live=QLabel('● LIVE');live.setStyleSheet(f'color:{GREEN}');h.addWidget(live);o.addWidget(self.head);self.image=CameraImage();o.addWidget(self.image,1);self.foot=QWidget();f=QHBoxLayout(self.foot);f.setContentsMargins(8,2,8,2);self.people=QLabel('0 People');self.fps=QLabel('-- FPS');f.addWidget(self.people);f.addStretch();f.addWidget(self.fps);o.addWidget(self.foot)
    def metrics(self,n,fps):self.people.setText(f'{n} '+('Person' if n==1 else 'People'));self.fps.setText(f'{fps:.0f} FPS' if fps>0 else '-- FPS')
    def camera_only(self,on):self.head.setVisible(not on);self.foot.setVisible(not on);self.setStyleSheet('border:0;background:#000' if on else '')

class Sidebar(QFrame):
    def __init__(self,change):
        super().__init__();self.setObjectName('sidebar');self.setFixedWidth(230);l=QVBoxLayout(self);brand=QLabel('◢  Apsidal');brand.setFont(font(25,QFont.Weight.DemiBold));l.addWidget(brand);l.addSpacing(14);self.buttons={}
        for i,(name,kind) in enumerate([('Live View','camera'),('People','person'),('Events','bell'),('Reports','other'),('Settings','other')]):b=QPushButton(name);b.setCheckable(True);b.setIcon(simple_icon(kind));b.setIconSize(QSize(22,22));b.setFixedHeight(54);b.clicked.connect(lambda checked=False,x=i:change(x));l.addWidget(b);self.buttons[i]=b
        l.addStretch();self.status=QLabel('System Status\nWaiting for realtime data');self.status.setAlignment(Qt.AlignCenter);self.status.setMinimumHeight(150);self.status.setObjectName('status');l.addWidget(self.status)
    def active(self,i):
        for k,b in self.buttons.items():b.setChecked(k==i)
    def update_live(self,state):
        if not state.get('connected'):self.status.setText('System Status\nML service offline');self.status.setStyleSheet(f'color:{RED}');return
        h=state.get('health') or {};res=h.get('service_resources') or {};self.status.setText(f"System Status\n{h.get('online',0)}/{h.get('total',6)} cameras online\nGPU {res.get('gpu_utilization_percent','—')}%");self.status.setStyleSheet(f'color:{GREEN}')

class Stat(QFrame):
    def __init__(self,name,color):super().__init__();self.setObjectName('stat');l=QVBoxLayout(self);self.value=QLabel('0');self.value.setFont(font(27,QFont.Weight.Bold));self.value.setStyleSheet(f'color:{color}');self.value.setAlignment(Qt.AlignCenter);n=QLabel(name);n.setAlignment(Qt.AlignCenter);l.addWidget(self.value);l.addWidget(n)

class RightRail(QWidget):
    def __init__(self):
        super().__init__();self.setFixedWidth(320);l=QVBoxLayout(self);cards=QFrame();cards.setObjectName('panel');g=QGridLayout(cards);self.total=Stat('Total People',BLUE);self.known=Stat('Known People',GREEN);self.unknown=Stat('Unknown People',ORANGE);self.cams=Stat('Active Cameras',CYAN);g.addWidget(self.total,0,0);g.addWidget(self.known,0,1);g.addWidget(self.unknown,1,0);g.addWidget(self.cams,1,1);l.addWidget(cards);recent=QFrame();recent.setObjectName('panel');rl=QVBoxLayout(recent);title=QLabel('Recent Views');title.setFont(font(17,QFont.Weight.DemiBold));rl.addWidget(title);self.recent_box=QVBoxLayout();rl.addLayout(self.recent_box);rl.addStretch();l.addWidget(recent,1)
    def update_live(self,state,recent):
        glob=((((state.get('reid') or {}).get('state') or {}).get('global')) or {});active=[gid for gid,v in glob.items() if v.get('active_tracks')];self.total.value.setText(str(len(active)));self.known.value.setText('0');self.unknown.value.setText(str(len(active)));h=state.get('health') or {};self.cams.value.setText(f"{h.get('online',0)}/{h.get('total',6)}")
        while self.recent_box.count():x=self.recent_box.takeAt(0).widget();x.deleteLater() if x else None
        for e in recent[:8]:x=QLabel(f"{e['gid']}\n{e['camera']} · {e['time']}");x.setStyleSheet(f'background:{CARD};padding:7px;border-radius:5px');self.recent_box.addWidget(x)

class LivePage(QWidget):
    def __init__(self):
        super().__init__();l=QVBoxLayout(self);l.setContentsMargins(0,0,0,0);self.head=QWidget();h=QHBoxLayout(self.head);title=QLabel('Live View');title.setFont(font(27,QFont.Weight.DemiBold));h.addWidget(title);h.addStretch();self.full=QPushButton('⛶');self.full.setFixedSize(42,42);h.addWidget(self.full);l.addWidget(self.head);self.grid=QGridLayout();self.grid.setSpacing(8);self.tiles={}
        for i,c in enumerate(CAMERAS):t=CameraTile(c,i+1);self.tiles[c]=t;self.grid.addWidget(t,i//2,i%2)
        for r in range(3):self.grid.setRowStretch(r,1)
        for c in range(2):self.grid.setColumnStretch(c,1)
        l.addLayout(self.grid,1)
    def camera_only(self,on):self.head.setVisible(not on);self.grid.setSpacing(2 if on else 8);[t.camera_only(on) for t in self.tiles.values()]

def item(v):x=QTableWidgetItem(str(v));x.setFlags(x.flags()&~Qt.ItemIsEditable);return x
class PeoplePage(QWidget):
    def __init__(self):super().__init__();l=QVBoxLayout(self);t=QLabel('People');t.setFont(font(27,QFont.Weight.DemiBold));l.addWidget(t);self.table=QTableWidget(0,5);self.table.setHorizontalHeaderLabels(['GLOBAL ID','CAMERAS','OBSERVATIONS','ROOM','STATUS']);self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.table.verticalHeader().hide();self.table.setObjectName('table');l.addWidget(self.table,1)
    def update_live(self,state):
        glob=((((state.get('reid') or {}).get('state') or {}).get('global')) or {});rows=[]
        for gid,v in glob.items():
            a=v.get('active_tracks') or {}
            if a:rows.append([gid,', '.join(sorted(a)),v.get('observations',0),', '.join(v.get('active_rooms') or []),'Active'])
        self.table.setRowCount(len(rows))
        for r,row in enumerate(rows):
            for c,v in enumerate(row):self.table.setItem(r,c,item(v))
class EventsPage(QWidget):
    def __init__(self):super().__init__();l=QVBoxLayout(self);t=QLabel('Events');t.setFont(font(27,QFont.Weight.DemiBold));l.addWidget(t);self.table=QTableWidget(0,5);self.table.setHorizontalHeaderLabels(['TIME','EVENT','CAMERA','IDENTITY','DETAILS']);self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.table.verticalHeader().hide();self.table.setObjectName('table');l.addWidget(self.table,1)
    def update_live(self,events):
        self.table.setRowCount(len(events))
        for r,e in enumerate(events):
            sim=e.get('similarity');detail=e.get('reason','detected')+(f' · {sim:.3f}' if isinstance(sim,(int,float)) else '')
            for c,v in enumerate([e['time'],'Person detected',e['camera'],e['gid'],detail]):self.table.setItem(r,c,item(v))
class EmptyPage(QWidget):
    def __init__(self,title):super().__init__();l=QVBoxLayout(self);t=QLabel(title);t.setFont(font(27,QFont.Weight.DemiBold));l.addWidget(t);x=QLabel('No demo data. No realtime source is connected for this page yet.');x.setAlignment(Qt.AlignCenter);x.setStyleSheet(f'color:{MUTED}');l.addWidget(x,1)

class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.resize(1672,941);self.setMinimumSize(1100,700);self.camera_mode=False;root=QWidget();self.setCentralWidget(root);root_l=QHBoxLayout(root);root_l.setContentsMargins(0,0,0,0);root_l.setSpacing(0);self.side=Sidebar(self.set_page);root_l.addWidget(self.side);self.body=QWidget();bl=QVBoxLayout(self.body);bl.setContentsMargins(0,0,0,0);root_l.addWidget(self.body,1);self.top=QWidget();tl=QHBoxLayout(self.top);m=QPushButton();m.setIcon(simple_icon('menu'));m.clicked.connect(lambda:self.side.setVisible(not self.side.isVisible()));tl.addWidget(m);tl.addStretch();self.clock=QLabel();tl.addWidget(self.clock);bl.addWidget(self.top);self.content=QWidget();self.content_l=QHBoxLayout(self.content);self.content_l.setContentsMargins(14,6,14,14);bl.addWidget(self.content,1);self.stack=QStackedWidget();self.live=LivePage();self.people=PeoplePage();self.events=EventsPage();self.reports=EmptyPage('Reports');self.settings=EmptyPage('Settings');[self.stack.addWidget(p) for p in [self.live,self.people,self.events,self.reports,self.settings]];self.content_l.addWidget(self.stack,1);self.rail=RightRail();self.content_l.addWidget(self.rail);self.live.full.clicked.connect(self.toggle_camera_mode);self.readers={};self.seen={};self.counts={c:0 for c in CAMERAS};self.last_tick=time.monotonic()
        for c in CAMERAS:r=FrameReader(c);r.start();self.readers[c]=r;self.seen[c]=-1
        self.state=LiveState();self.state.start();self.rt=QTimer(self);self.rt.setTimerType(Qt.PreciseTimer);self.rt.timeout.connect(self.render);self.rt.start(20);self.it=QTimer(self);self.it.timeout.connect(self.update_info);self.it.start(500);self.ct=QTimer(self);self.ct.timeout.connect(self.update_clock);self.ct.start(1000);self.update_clock();self.set_page(0);self.theme()
    def theme(self):self.setStyleSheet(f"QMainWindow,QWidget{{background:{BG};color:{TEXT}}}#sidebar{{background:{SIDEBAR};border-right:1px solid {BORDER}}}#sidebar QPushButton{{text-align:left;padding-left:15px;border-radius:7px;border:0}}#sidebar QPushButton:checked{{background:{BLUE}}}#status,#panel,#cameraTile,#stat{{background:{PANEL};border:1px solid {BORDER};border-radius:7px}}QTableWidget#table{{background:{PANEL};border:1px solid {BORDER};color:{TEXT}}}QHeaderView::section{{background:#00142d;color:#c7d2e2;border:0;padding:7px}}")
    def set_page(self,i):self.stack.setCurrentIndex(i);self.side.active(i)
    def update_clock(self):self.clock.setText(datetime.now().strftime('%H:%M:%S   %d %b %Y'))
    def render(self):
        for c,r in self.readers.items():
            img,v=r.latest()
            if img is not None and v>self.seen[c]:self.seen[c]=v;self.live.tiles[c].image.set_frame(img)
    def update_info(self):
        state,recent,events=self.state.snapshot();now=time.monotonic();dt=max(.1,now-self.last_tick);self.last_tick=now;dets=((state.get('detections') or {}).get('cameras') or {})
        for c,r in self.readers.items():cur=r.frames;fps=(cur-self.counts[c])/dt;self.counts[c]=cur;self.live.tiles[c].metrics(len((dets.get(c) or {}).get('boxes') or []),fps)
        self.side.update_live(state);self.rail.update_live(state,recent);self.people.update_live(state);self.events.update_live(events)
    def toggle_camera_mode(self):
        self.camera_mode=not self.camera_mode
        if self.camera_mode:self.set_page(0);self.side.hide();self.top.hide();self.rail.hide();self.content_l.setContentsMargins(0,0,0,0);self.content_l.setSpacing(0);self.live.camera_only(True);self.showFullScreen()
        else:self.live.camera_only(False);self.side.show();self.top.show();self.rail.show();self.content_l.setContentsMargins(14,6,14,14);self.content_l.setSpacing(6);self.showMaximized()
    def keyPressEvent(self,e):
        if e.key()==Qt.Key_Escape and self.camera_mode:self.toggle_camera_mode();return
        super().keyPressEvent(e)
    def closeEvent(self,e):self.state.stop();[r.stop() for r in self.readers.values()];e.accept()

def run():app=QApplication.instance() or QApplication([]);app.setStyle('Fusion');w=DashboardWindow();w.showMaximized();return app.exec()
