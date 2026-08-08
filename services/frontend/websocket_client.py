import json
from PySide6.QtCore import QObject,QTimer,Signal,QUrl
from PySide6.QtWebSockets import QWebSocket
from PySide6.QtNetwork import QAbstractSocket

class WebSocketClient(QObject):
    message=Signal(dict);connected=Signal();disconnected=Signal();error=Signal(str)
    def __init__(self,url="ws://127.0.0.1:8000/ws",parent=None):
        super().__init__(parent);self.url=QUrl(url);self.socket=QWebSocket();self.delay=1000
        self.timer=QTimer(self);self.timer.setSingleShot(True);self.timer.timeout.connect(self.connect)
        self.socket.connected.connect(self._connected);self.socket.disconnected.connect(self._disconnected)
        self.socket.textMessageReceived.connect(self._message);self.socket.errorOccurred.connect(lambda _e:self.error.emit(self.socket.errorString()))
    def connect(self):self.socket.open(self.url)
    def close(self):
        self.timer.stop()
        if self.socket.state()==QAbstractSocket.ConnectedState:self.socket.close()
        else:self.socket.abort()
    def _connected(self):self.delay=1000;self.connected.emit()
    def _disconnected(self):self.disconnected.emit();self.timer.start(self.delay);self.delay=min(30000,self.delay*2)
    def _message(self,text):
        try:self.message.emit(json.loads(text))
        except ValueError:self.error.emit("Invalid WebSocket message")
