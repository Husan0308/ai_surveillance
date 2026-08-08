import math,time

class LiveDecay:
    def __init__(self,enabled=True,half_life_seconds=30):self.enabled=enabled;self.half_life=max(.001,float(half_life_seconds));self.last=time.time()
    def apply(self,grid,timestamp):
        elapsed=max(0,timestamp-self.last);self.last=max(self.last,timestamp)
        if self.enabled and elapsed:grid*=math.exp(-math.log(2)*elapsed/self.half_life)
