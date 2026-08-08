"""Canonical single-instance OSNet batch embedding model."""
import threading
import time
import cv2
import numpy as np

class OSNetReIDModel:
    def __init__(self,config):
        import torch,torchreid
        cfg=config.get("ai",{}).get("reid",{});self.torch=torch;requested=str(cfg.get("device","cpu"))
        self.device=torch.device("cuda:0" if requested=="auto" and torch.cuda.is_available() else requested)
        name=str(cfg.get("model","osnet_x0_25"));checkpoint=str(cfg.get("checkpoint","models/osnet_x0_25_market1501.pt"))
        self.model=torchreid.models.build_model(name=name,num_classes=751,loss="softmax",pretrained=False)
        state=torch.load(checkpoint,map_location="cpu");state=state.get("state_dict",state);state={k.replace("module.",""):v for k,v in state.items()}
        self.model.load_state_dict(state,strict=False);self.model.eval().to(self.device);self.lock=threading.Lock();self.model_identity=id(self.model)
        self.mean=np.array([.485,.456,.406],np.float32);self.std=np.array([.229,.224,.225],np.float32)

    def extract_batch(self,crops):
        started=time.perf_counter()
        valid=[];indices=[];output=[None]*len(crops)
        for index,crop in enumerate(crops):
            if crop is None or crop.size==0:continue
            rgb=cv2.cvtColor(cv2.resize(crop,(128,256)),cv2.COLOR_BGR2RGB);valid.append(rgb);indices.append(index)
        if not valid:return output,{"gpu_ms":0}
        array=np.asarray(valid,np.float32)/255;array=np.ascontiguousarray(((array-self.mean)/self.std).transpose(0,3,1,2))
        with self.lock,self.torch.inference_mode():
            # torchreid OSNet returns embeddings directly while in eval mode.
            tensor=self.torch.from_numpy(array).to(self.device);features=self.model(tensor)
            if isinstance(features,(tuple,list)):features=features[0]
            features=features.reshape(features.shape[0],-1);features=features/(features.norm(dim=1,keepdim=True)+1e-8);host=features.cpu().numpy()
        for row,index in enumerate(indices):output[index]=host[row]
        return output,{"gpu_ms":(time.perf_counter()-started)*1000}

    def close(self):self.model=None
