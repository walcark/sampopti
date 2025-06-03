############################################################################
# Loading useful modules                                                   #
############################################################################
from typing import List, Tuple, Callable
from scipy.interpolate import interp1d
import pycuda.driver as cuda
import numpy as np
import functools
import logging
logger = logging.getLogger(__name__)

from sampopti.classes import SunSat
from sampopti.extlibs import (Smartg, 
                              Sensor, 
                              MLUT, 
                              AtmAFGL, 
                              LambSurface,
                              Environment)


############################################################################
# Test shapes                                                              #
############################################################################
def gaussian(
    coords: List[Tuple], 
    sigma: float
) -> float:
    """Simple gaussian function for test purpose."""
    radius = [np.sqrt(x**2 + y**2) for (x, y) in coords]
    return [np.exp(-(r / sigma)**2) for r in radius]


def smooth_disk(
    coords: List[Tuple], 
    rmax: float, 
    smooth_factor: float = 10.0
) -> float:
    """Simple smooth disk function for test purpose."""
    radius = [np.sqrt(x**2 + y**2) for (x, y) in coords]
    return [1 / (1 + np.exp(smooth_factor * (r - rmax))) for r in radius]


############################################################################
# Decorator to handle cuda context                                         #
############################################################################
def with_cuda_context(func: Callable) -> Callable:
    """
    A decorator that creates a CUDA context before calling 
    a given function 'func' and then cleans the context.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cuda.init()
        ctx = cuda.Device(0).make_context()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error("CUDA computation failed: %s", str(e))
            raise
        finally:
            ctx.pop()
            ctx.detach()
        return result
    return wrapper 


############################################################################
# Calculation of RHO_TOA                                                   #
############################################################################
@with_cuda_context
def rho_toa(
    points: List[Tuple],
    wavelength: float,
    atmosphere: AtmAFGL,
    surface: LambSurface,
    environment: Environment,
    sunsat: SunSat = SunSat(), 
    nb_photons: int = 1E5, 
    nb_angles: int = 1E4, 
    **kwargs
) -> np.ndarray:
    """
    Calculates the Top Of Atmosphere reflectance (TOA) for the given 
    instances of Ground in params.GROUND. The TOA reflectance can 
    either be with rhoAtm (RMIN = 0) or without (RMIN = 1).
    """
    logger.debug("Calling rho_toa with %d points", len(points))
    # Grid of satellite positions
    x_sat, y_sat = sunsat.satellite_relative_position
    sat_locs = [(x + x_sat, y + y_sat) for (x, y) in points]

    # Total number of photons to launch
    nb_photons_tot: int = len(sat_locs) * nb_photons

    # Top-of-atmosphere sensors to launch photons
    sensors_toa: List[Sensor] = [
        Sensor(
            POSX=pos_x, 
            POSY=pos_y, 
            POSZ=sunsat.sat_height,
            THDEG=180-sunsat.vza_deg,
            PHDEG=sunsat.vaa_deg,
            LOC='ATMOS'
        ) for (pos_x, pos_y) in sat_locs
    ]
    
    # Computation with Smart-G and extraction of e_coupling
    sg = Smartg(back=True)
    mlut: MLUT = sg.run(
        wl=wavelength, 
        atm=atmosphere,
        surf=surface, 
        env=environment,
        sensor=sensors_toa,
        le=sunsat.sun_le,
        NBPHOTONS=nb_photons_tot,
        NF=nb_angles,
    )
    logger.info("Finished rho_toa computation")
    return list(mlut["I_up (TOA)"][:, 0])


############################################################################
# Calculation of RHO_COUPLING                                              #
############################################################################
@with_cuda_context
def rho_coupling(
    points: List[Tuple],
    wavelength: float,
    atmosphere: AtmAFGL,
    surface: LambSurface,
    environment: Environment,
    sunsat: SunSat = SunSat(),
    nb_photons: int = 1E5, 
    nb_angles: int = 1E4, 
    **kwargs
) -> np.ndarray:
    """
    Direct calculation of 'E_coupling' for a given atmosphere and 
    environment on a 2D grid.
    """
    logger.debug("Calling rho_coupling with %d points", len(points))

    # 1. Keeping only unique points
    r = np.sqrt(np.array([x**2 + y**2 for (x, y) in points]))
    r_unique = r[np.unique(np.round(r, 0), return_index=True)[1]]
    r_unique = np.append(0.0, r_unique)

    # 2. Simulation on single coordinate
    lamb_emit_boa: List[Sensor] = [
        Sensor(
            POSX=0.0, 
            POSY=py, 
            POSZ=0.0,
            FOV=90.,
            TYPE=1,
            LOC='ATMOS'
        ) for py in r_unique
    ]

    mlut: MLUT = Smartg(back=True).run(
        wl=wavelength,
        atm=atmosphere,
        surf=surface,
        env=environment,
        sensor=lamb_emit_boa,
        le=sunsat.sun_le,
        NBPHOTONS=len(r_unique) * nb_photons,
        NF=nb_angles,
    )
    e_coupling: np.ndarray = mlut["I_up (TOA)"][:, 0]

    # 3. Reinterpolating in full-coordinates
    interp_func = interp1d(r_unique, 
                           e_coupling, 
                           kind='linear', 
                           fill_value="extrapolate")
    e_coupling = interp_func(r)
    logger.info("Finished rho_coupling computation")
    return e_coupling


