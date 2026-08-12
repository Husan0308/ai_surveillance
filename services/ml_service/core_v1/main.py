import uvicorn
from .app import app,core_cfg

if __name__=='__main__':
    uvicorn.run(app,host=str(core_cfg.get('ml_host','0.0.0.0')),port=int(core_cfg.get('ml_port',8001)),log_level='info')
