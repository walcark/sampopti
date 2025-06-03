############################################################################
# Loading useful modules                                                   #
############################################################################
from typing import List, Callable, Any, Tuple, Dict
import numpy as np
import logging
logger = logging.getLogger(__name__)

from sampopti.classes import Pixel


############################################################################
# Main methods for quadtree optimization                                   #
############################################################################
def deduplicate_evaluate_and_remap(
    pixels: List[Pixel], 
    func: Callable[..., List[float]],
    cache: Dict[Tuple[float, float], float],
    **kwargs
) -> List[bool]:
    """
    Identify the unique coordinates for the list of pixels, then call
    func for each unique coordinate that has not been evaluated before.
    Remap the results on each pixel, then decides subdivision or not.
    """
    logger.debug("Evaluating %d pixels", len(pixels))
    unique_coords, remapped = [], []
    coord_to_index = {}
    coord_groups = [pixel.get_points for pixel in pixels]

    for group in coord_groups:
        remapped_group = []
        for coord in group:
            if coord not in coord_to_index:
                coord_to_index[coord] = len(unique_coords)
                unique_coords.append(coord)
            remapped_group.append(coord_to_index[coord])
        remapped.append(tuple(remapped_group))

    # Identifier uniquement les coordonnées à évaluer
    coords_to_eval = [c for c in unique_coords if c not in cache]
    logger.debug("→ %d unique coordinates to evaluate", len(coords_to_eval))
    if coords_to_eval:
        new_values = func(coords_to_eval, **kwargs)
        if len(new_values) != len(coords_to_eval):
            logger.error("Function returned %d values for %d coords", 
                         len(new_values), 
                         len(coords_to_eval))
        for coord, val in zip(coords_to_eval, new_values):
            cache[coord] = val

    # Reconstruction des valeurs pour tous les pixels
    unique_values = [cache[c] for c in unique_coords]
    values = [tuple(unique_values[i] for i in group) for group in remapped]

    # Vérification de la subdivision
    subdivision = [pix.check_subdivide(val) 
                   for (pix, val) in zip(pixels, values)]
    return subdivision


def recursive_subdivision(
    test_pixels: List[Pixel],
    final_pixels: List[Pixel],
    func: Callable[..., Any],
    cache: Dict[Tuple[float, float], float],
    **kwargs
) -> List[Pixel]:
    """
    Recursive subdivision of the pixels. The algorithm is cached
    to avoid re-computation of values during the QuadTree 
    evolution.
    """
    new_test_pixels = []
    check_subdivision = deduplicate_evaluate_and_remap(test_pixels, 
                                                       func, 
                                                       cache, 
                                                       **kwargs)
    
    logger.debug("Subdivision step → %d new test pixels", len(new_test_pixels))

    for is_pix_subdiv, pixel in zip(check_subdivision, test_pixels):
        if is_pix_subdiv:
            new_test_pixels.extend(pixel.get_subdivision())
        else:
            final_pixels.append(pixel)

    if not new_test_pixels:
        logger.warning("No more subdivisions; reached final resolution.")
        return final_pixels

    return recursive_subdivision(new_test_pixels, 
                                 final_pixels, 
                                 func, 
                                 cache, 
                                 **kwargs)


def pixels_to_centers(pixels: List[Pixel]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns two numpy array (X, Y) or the pixels center coordinates.
    """
    loc, val = [], []
    for pixel in pixels:
        loc.append([pixel.mx, pixel.my])
        val.append(pixel.values[4])  # valeur au centre
    return np.array(loc), np.array(val)
