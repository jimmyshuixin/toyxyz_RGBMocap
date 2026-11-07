"""Bound preview textures while retaining every original image in the rest asset."""
import numpy as np

IMAGE_BYTES = 256 * 1024**2
TOTAL_BYTES = 512 * 1024**2


def allocate_preview_sizes(original, image_bytes=IMAGE_BYTES, total_bytes=TOTAL_BYTES):
    """Halve the largest texture until both RGBA8 budgets fit, deterministically.

    Integer power-of-two factors and rounded-up dimensions preserve edge texels
    for odd image sizes. Images already within the shared budget stay unchanged.
    """
    sizes, factors = {}, {}
    for key, size in original.items():
        if len(size) != 2 or any(not isinstance(v, int) or v < 1 for v in size):
            raise ValueError('Invalid original character texture dimensions')
        sizes[key] = list(size)
        factors[key] = 1
    if image_bytes < 4 or total_bytes < 4 * len(sizes):
        raise ValueError('Character texture preview budget cannot fit even 1x1 images')
    while sizes:
        costs = {key: w*h*4 for key, (w, h) in sizes.items()}
        if max(costs.values()) <= image_bytes and sum(costs.values()) <= total_bytes:
            break
        key = max(costs, key=costs.get)
        factors[key] *= 2
        sizes[key] = [(v + factors[key]-1)//factors[key] for v in original[key]]
    return sizes, factors


def area_reduce(pixels, factor):
    """Average complete source texel blocks, including partial right/bottom edges."""
    if factor == 1:
        return pixels
    height, width = pixels.shape[:2]
    xs, ys = np.arange(0, width, factor), np.arange(0, height, factor)
    result = np.add.reduceat(np.add.reduceat(pixels, xs, axis=1), ys, axis=0)
    counts = np.minimum(factor, height-ys)[:, None] * np.minimum(factor, width-xs)[None, :]
    return result / counts[..., None]
