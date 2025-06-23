# Sampopti
Sampopti is a plug-in aiming to perform fast MCMC simulation of physical quantities such as Top-Of-Atmosphere reflectance ($\rho_\mathrm{toa}$) within the Smart-G software (https://github.com/hygeos/smartg). The idea is to optimally sample the quantity on a given area based its variations. This optimal sample is based on a QuadTree-like approach, adapted to the use of Smart-G.

# Quadtree approach 
Given a geographical area defined by its three corners ... . TBR

# Usage
python -m sampopti.skeleton   --func rho_toa --atm afglms --aerosol-type continental_average --aod 0.15 --ref-wl 550 --albedo 1.0 --env 1 --env-size 10 --extent 30 --low-res 0.5 --high-res 10 --threshold 0.05 --output result.png --show -vv --interpolate --interp-method gaussian
