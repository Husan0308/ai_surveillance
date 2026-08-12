"""Small CPU constant-velocity Kalman filter for bounding boxes."""
import numpy as np

class BoxMotionModel:
    def __init__(self, bbox):
        cx, cy, width, height = self._measurement(bbox)
        self.state = np.array([cx, cy, width, height, 0, 0, 0, 0], np.float32)
        self.covariance = np.eye(8, dtype=np.float32) * 10
        self.observation = np.zeros((4, 8), np.float32); self.observation[:, :4] = np.eye(4, dtype=np.float32)
        self.process_noise = np.eye(8, dtype=np.float32) * .05
        self.measurement_noise = np.eye(4, dtype=np.float32)
        self.bbox = np.asarray(bbox, np.float32)

    @staticmethod
    def _measurement(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def _bbox(state):
        cx, cy, width, height = state[:4]
        return np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], np.float32)

    @property
    def velocity(self): return self.state[4:6]

    @staticmethod
    def _transition(dt):
        transition=np.eye(8,dtype=np.float32);transition[:4,4:]=np.eye(4,dtype=np.float32)*max(0.0,float(dt));return transition

    def predict(self,dt=0.0):
        predicted = self._transition(dt) @ self.state
        return self._bbox(predicted)

    def _advance(self,dt):
        transition=self._transition(dt);self.state=transition @ self.state
        self.covariance=transition @ self.covariance @ transition.T+self.process_noise*max(.01,float(dt))

    def update(self,bbox,dt=1.0):
        self._advance(dt); measurement = np.asarray(self._measurement(bbox), np.float32)
        residual = measurement - self.observation @ self.state
        innovation = self.observation @ self.covariance @ self.observation.T + self.measurement_noise
        gain = np.linalg.solve(innovation.T, (self.covariance @ self.observation.T).T).T
        self.state += gain @ residual
        self.covariance = (np.eye(8, dtype=np.float32) - gain @ self.observation) @ self.covariance
        self.bbox = self._bbox(self.state)

    def miss(self,dt=1.0):
        self._advance(dt); self.bbox = self._bbox(self.state)
