import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
import sys

from sampopti.extlibs import AtmAFGL, AerOPAC, LambSurface, Albedo_cst, Environment
from sampopti.functions import gaussian, rho_toa, smooth_disk, rho_coupling
from sampopti.optimal import recursive_subdivision, Pixel
from sampopti.plotter import show_quadtree_pixels
from sampopti.logging_utils import setup_logging
from sampopti.classes import SunSat


logger = logging.getLogger(__name__)

__author__ = "Kévin Walcarius"
__copyright__ = "Kévin Walcarius"
__license__ = "MIT"

def parse_args(args):
    parser = argparse.ArgumentParser(description="Run adaptive subdivision with atmospheric or test functions.")

    parser.add_argument("--func", choices=["gaussian", "smooth_disk", "rho_toa", "rho_coupling"],
                        default="gaussian", help="Function to evaluate")
    parser.add_argument("--extent", type=float, default=100.0, help="Half size of the domain (default: 100.0)")
    parser.add_argument("--low-res", type=float, default=1.0, help="Minimal pixel resolution (default: 1.0)")
    parser.add_argument("--high-res", type=float, default=50.0, help="Maximal pixel resolution (default: 50.0)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Subdivision threshold (default: 0.05)")

    parser.add_argument("--sza", type=float, default=30.0, help="Solar Zenith Angle (SZA) in degrees")
    parser.add_argument("--saa", type=float, default=0.0, help="Solar Azimuth Angle (SAA) in degrees")
    parser.add_argument("--vza", type=float, default=0.0, help="View Zenith Angle (VZA) in degrees")
    parser.add_argument("--vaa", type=float, default=0.0, help="View Azimuth Angle (VAA) in degrees")
    parser.add_argument("--height", type=float, default=700.0, help="Satellite height (km)")

    parser.add_argument("--wavelength", type=float, default=443.0, help="Wavelength (nm) used in rho_toa")
    parser.add_argument("--atm", default="afglms", help="Atmospheric profile filename (default: afglms)")
    parser.add_argument("--aerosol-type", default="continental_average", help="Aerosol model name (Smart-G)")
    parser.add_argument("--aod", type=float, default=0.2, help="Aerosol Optical Depth at reference wl")
    parser.add_argument("--ref-wl", type=float, default=590.0, help="AOD reference wavelength (nm)")
    parser.add_argument("--albedo", type=float, default=1.0, help="Lambertian surface albedo")
    parser.add_argument("--env-size", type=float, default=30.0, help="Environment size (km)")
    parser.add_argument("--env", type=int, default=2, help="Environment type (ENV)")
    parser.add_argument("--x0", type=float, default=0.0, help="X center of environment")
    parser.add_argument("--y0", type=float, default=0.0, help="Y center of environment")
    parser.add_argument("--nb-photons", type=int, default=int(1e5), help="Number of photons per evaluation")

    parser.add_argument("--interpolate", action="store_true", help="Interpolate onto a regular grid")
    parser.add_argument("--grid-res", type=float, default=0.1, help="Resolution of interpolation grid (km)")
    parser.add_argument("--interp-method", choices=["knn", "ct", "idw", "gaussian"], default="knn", help="Interpolation method")
    parser.add_argument("--k-neighbors", type=int, default=4, help="Number of neighbors for KNN/IDW/Gaussian interpolation")
    parser.add_argument("--save-grid", type=str, help="Filename to save interpolated grid as GeoTIFF")

    parser.add_argument("--show", action="store_true", help="Display result")
    parser.add_argument("--output", type=str, help="Save output as PNG")
    parser.add_argument("-v", "--verbose", action="store_const", const=logging.INFO, dest="loglevel")
    parser.add_argument("-vv", "--very-verbose", action="store_const", const=logging.DEBUG, dest="loglevel")
    parser.set_defaults(loglevel=logging.WARNING)

    return parser.parse_args(args)

def main(args):
    setup_logging(loglevel=args.loglevel)
    logger.setLevel(args.loglevel)

    lim = args.extent
    root = Pixel(-lim, lim, -lim, lim, 0, 0, 0, args.threshold)

    cache = {}
    func_kwargs = {}

    if args.func == "gaussian":
        func = gaussian
        func_kwargs = {"sigma": 40.0}

    elif args.func == "smooth_disk":
        func = smooth_disk
        func_kwargs = {"rmax": 50.0, "smooth_factor": 10.0}

    elif args.func == "rho_toa" or args.func == "rho_coupling":
        sunsat = SunSat(args.sza, args.saa, args.vza, args.vaa, args.height)
        atm = AtmAFGL(args.atm, [AerOPAC(args.aerosol_type, args.aod, args.ref_wl)],
                      grid=np.linspace(100., 0., 101),
                      pfgrid=np.array([100., 20., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.]))
        surface = LambSurface(ALB=Albedo_cst(args.albedo))
        environment = Environment(ENV=args.env, ENV_SIZE=args.env_size, X0=args.x0, Y0=args.y0)
        func = rho_toa if args.func == "rho_toa" else rho_coupling
        func_kwargs = dict(wavelength=args.wavelength,
                           nb_photons=args.nb_photons,
                           sunsat=sunsat,
                           atmosphere=atm,
                           surface=surface,
                           environment=environment)

    min_depth = int(np.ceil(np.log2((2 * lim) / args.high_res)))
    max_depth = int(np.ceil(np.log2((2 * lim) / args.low_res)))
    root.min_depth = min_depth
    root.max_depth = max_depth

    pixels = recursive_subdivision([root], [], func, cache=cache, **func_kwargs)
    show_quadtree_pixels(pixels)
    plt.show()


def run():
    main(parse_args(sys.argv[1:]))

if __name__ == "__main__":
    run()
