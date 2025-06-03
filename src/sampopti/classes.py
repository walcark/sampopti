############################################################################
# Imports                                                                  #
############################################################################
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import logging
logger = logging.getLogger(__name__)

from adjeffcorr.external_libs import Sensor

############################################################################
# SunSat class                                                             #
############################################################################
@dataclass 
class SunSat: 
    sza_deg: float = 0.0
    saa_deg: float = 0.0
    vza_deg: float = 0.0
    vaa_deg: float = 0.0
    sat_height: float = 700.0

    @property
    def sun_le(self) -> Dict[str, float]:
        return {"th_deg": self.sza_deg, "phi_deg": self.saa_deg, "zip": True}
    
    @property
    def sat_le(self) -> Dict[str, float]:
        return {"th_deg": self.vza_deg, "phi_deg": self.vaa_deg, "zip": True}
    
    @property
    def sun_sensor(self) -> Sensor:
        return Sensor(
            POSZ=self.sat_height, 
            THDEG=180.0 - self.sza_deg, 
            PHDEG=self.saa_deg, 
            LOC='ATMOS'
        )

    @property
    def sat_sensor(self) -> Sensor:
        return Sensor(
            POSZ=self.sat_height, 
            THDEG=180.0 - self.vza_deg, 
            PHDEG=self.vaa_deg, 
            LOC='ATMOS'
        )                      
    
    @property
    def satellite_relative_position(self) -> Tuple[float, float]:
        """
        Returns the relative position (x, y) of a satellite from 
        the point P(x=0, y=0, z=0) it is looking at on the earth.
        """
        tan_vza: float = np.tan(np.radians(self.vza_deg))
        cos_vaa: float = np.cos(np.radians(180 - self.vaa_deg))
        sin_vaa: float = np.sin(np.radians(180 - self.vaa_deg))

        x: float  = (self.sat_height * tan_vza) * cos_vaa
        y: float = (self.sat_height * tan_vza) * sin_vaa
        logger.debug("Computed satellite relative position: (%.2f, %.2f)", x, y)

        return (np.round(x, 4), np.round(y, 4))
    
    def __str__(self) -> str:
        return "-".join([
            f"SZA-{self.sza_deg}", f"SAA-{self.saa_deg}",
            f"VZA-{self.vza_deg}", f"VAA-{self.vaa_deg}",
            f"H-{self.sat_height}"
        ])
