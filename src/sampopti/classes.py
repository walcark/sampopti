############################################################################
# Loading useful modules                                                   #
############################################################################
from typing import Dict, Tuple, List 
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
        return self._le(self.sza_deg, self.saa_deg)
    
    @property
    def sat_le(self) -> Dict[str, float]:
        return self._le(self.vza_deg, self.vaa_deg)
    
    @property
    def sun_sensor(self) -> Sensor:
        return self._sensor(self.sat_height, self.sza_deg, self.saa_deg)

    @property
    def sat_sensor(self) -> Sensor:
        return self._sensor(self.sat_height, self.vza_deg, self.vaa_deg)

    def _le(self, za: float, aa: float) -> Dict[str: float]:
        return {"th_deg": za, "phi_deg": aa, "zip": True}

    def _sensor(self, height: float, za: float, aa: float) -> Sensor:
        return Sensor(POSZ=height, THDEG=180.0 - za, PHDEG=aa, LOC='ATMOS')                   
    
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
        logger.debug("Computed satellite relative position: (%.2f, %.2f)", 
                     x, y)
        return (np.round(x, 4), np.round(y, 4))
    
    def __str__(self) -> str:
        return "-".join([
            f"SZA-{self.sza_deg}", f"SAA-{self.saa_deg}",
            f"VZA-{self.vza_deg}", f"VAA-{self.vaa_deg}",
            f"H-{self.sat_height}"
        ])



############################################################################
# Pixel class (for QuadTree like operations)                               #
############################################################################
@dataclass
class Pixel:
    """ 
    Equivalent to a Quadratic Tree without 
    recursivity operations.
    """
    x0: float
    x1: float
    y0: float
    y1: float
    depth: int
    min_depth: int
    max_depth: int
    threshold: float

    def __post_init__(self):
        self.mx = 0.5 * (self.x0 + self.x1)
        self.my = 0.5 * (self.y0 + self.y1)

    @property
    def get_points(self) -> List[Tuple[float, float]]:
        """
        Returns the corners and the center of the pixel.
        """
        return [
            (self.x0, self.y0),
            (self.x1, self.y0),
            (self.x0, self.y1),
            (self.x1, self.y1),
            (self.mx, self.my)
        ]

    def check_subdivide(self, values: List[float]) -> bool:
        """
        Checks if the pixel needs a subdivision. If the maximum 
        difference between the evaluation points is over a given
        threshold, performs the subdivision.
        """
        self.values = values
        if self.depth < self.min_depth:
            logger.warning("Forced subdivision at depth=%d (min_depth=%d)", 
                           self.depth, 
                           self.min_depth)
            return True

        if self.depth < self.min_depth:
            return True
        max_diff = max(values) - min(values)
        return (self.depth < self.max_depth) and (max_diff > self.threshold)

    def get_subdivision(self) -> List["Pixel"]:
        """
        Returns the 4 sub-pixels of the current pixel instance.
        """
        args = dict(min_depth=self.min_depth,
                    max_depth=self.max_depth,
                    threshold=self.threshold,
                    depth=self.depth + 1)
        return [
            Pixel(self.x0, self.mx, self.y0, self.my, **args),
            Pixel(self.mx, self.x1, self.y0, self.my, **args),
            Pixel(self.x0, self.mx, self.my, self.y1, **args),
            Pixel(self.mx, self.x1, self.my, self.y1, **args)
        ]