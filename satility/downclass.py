from __future__ import annotations

import pathlib
from typing import Callable
from georeader.readers import probav_image_operational as probav
from georeader.readers import spotvgt_image_operational as spotvgt
from dataclasses import dataclass

@dataclass(frozen=True)
class SensorCfg:
    collection_id: str
    reader_factory: Callable[[pathlib.Path], object]

SENSORS = {
    "probav": SensorCfg(
        collection_id="urn:eop:VITO:PROBAV_L2A_1KM_HDF_V2",
        reader_factory=lambda p: probav.ProbaV(p, level_name="LEVEL2A"),
    ),
    "spot": SensorCfg(
        collection_id="urn:ogc:def:EOP:VITO:VGT_P",
        reader_factory=lambda p: spotvgt.SpotVGT(p),
    ),
}