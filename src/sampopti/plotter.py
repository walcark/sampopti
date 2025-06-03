############################################################################
# Loading useful modules                                                   #
############################################################################
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from typing import List

from sampopti.classes import Pixel


############################################################################
# QuadTree grid plot                                                       #
############################################################################
def show_quadtree_pixels(pixels: List[Pixel], ax: plt.Axes = None) -> None:
    """
    Affiche les rectangles des pixels d'une subdivision quadtree.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.set_title("QuadTree grid", fontsize="x-large", pad=15)
        ax.set_xlabel("X coordinate [units]", fontsize="x-large")
        ax.set_ylabel("Y coordinate [units]", fontsize="x-large")
        ax.set_aspect(1)

    rects = [
        patches.Rectangle(
            (pixel.x0, pixel.y0),
            pixel.x1 - pixel.x0,
            pixel.y1 - pixel.y0
        )
        for pixel in pixels
    ]

    collection = PatchCollection(
        rects, edgecolor='black', facecolor='none', lw=0.5
    )
    ax.add_collection(collection)

    xmins = [p.x0 for p in pixels]
    xmaxs = [p.x1 for p in pixels]
    ymins = [p.y0 for p in pixels]
    ymaxs = [p.y1 for p in pixels]
    ax.set_xlim(min(xmins), max(xmaxs))
    ax.set_ylim(min(ymins), max(ymaxs))
    ax.tick_params(axis='both', which="major", labelsize=14)

    if ax is None:
        fig.tight_layout()
        plt.show()