from dataclasses import dataclass
from enum import Enum
import numpy as np

class HeatmapMode(str,Enum): LIVE="LIVE";MINUTE="MINUTE";HOURLY="HOURLY";DAILY="DAILY"
class AccumulationMode(str,Enum): POINT="POINT";FOOTPRINT="FOOTPRINT"
@dataclass(frozen=True,slots=True)
class CameraPosition:
    camera_id:str;frame_id:int;identity_key:str;x_norm:float;y_norm:float;frame_width:int;frame_height:int;timestamp:float
@dataclass(frozen=True,slots=True)
class CameraHeatmapSnapshot:
    camera_id:str;timestamp:float;mode:HeatmapMode;grid_width:int;grid_height:int;frame_width:int;frame_height:int;max_value:float;values:np.ndarray
