import itertools

def fix_lastchunk(iterchunks, s2dim, chunk_size):
    """
    Fix the last chunk of the overlay to ensure it aligns with image boundaries.

    Args:
        iterchunks (list): List of chunks created by itertools.product.
        s2dim (tuple): Dimension of the S2 images.
        chunk_size (int): Size of the chunks.

    Returns:
        list: List of adjusted chunk coordinates.
    """
    itercontainer = []

    for index_i, index_j in iterchunks:
        # Adjust if the chunk extends beyond bounds
        if index_i + chunk_size > s2dim[0]:
            index_i = max(s2dim[0] - chunk_size, 0)
        if index_j + chunk_size > s2dim[1]:
            index_j = max(s2dim[1] - chunk_size, 0)

        itercontainer.append((index_i, index_j))

    return itercontainer


def define_iteration(dimension: tuple, chunk_size: int, overlap: int = 0):
    """
    Define the iteration strategy to walk through the image with an overlap.

    Args:
        dimension (tuple): Dimension of the S2 image.
        chunk_size (int): Size of the chunks.
        overlap (int): Size of the overlap between chunks.

    Returns:
        list: List of chunk coordinates.
    """
    dimy, dimx = dimension

    if chunk_size > max(dimx, dimy):
        return [(0, 0)]

    # Adjust step to create overlap
    y_step = chunk_size - overlap
    x_step = chunk_size - overlap

    # Generate initial chunk positions
    iterchunks = list(itertools.product(range(0, dimy, y_step), range(0, dimx, x_step)))

    # Fix chunks at the edges to stay within bounds
    iterchunks_fixed = fix_lastchunk(
        iterchunks=iterchunks, s2dim=dimension, chunk_size=chunk_size
    )

    return iterchunks_fixed



def compute_valid_roi(
    row_off: int,
    col_off: int,
    *,
    chunk_size: int,
    overlap: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int, int, int]:
    """
    Compute the valid Region-Of-Interest (ROI) within an overlapped tile.

    Returns
    -------
    (offset_x, offset_y, length_x, length_y, sub_x_start, sub_y_start)

    * offset_x, offset_y – upper-left corner where the ROI should be written
      in the **final** (global) image.
    * length_x, length_y – width and height of the ROI that will be written.
    * sub_x_start, sub_y_start – where the ROI starts inside the **local**
      tile read from disk.

    Notes
    -----
    The logic keeps `overlap // 2` pixels of overlap between interior tiles
    but uses the full tile at the right and bottom borders.
    """
    # Destination offset in the global image
    offset_x = 0 if col_off == 0 else col_off + overlap // 2
    offset_y = 0 if row_off == 0 else row_off + overlap // 2

    # Width of the valid area
    if offset_x + chunk_size == width:        # touches right edge
        length_x = chunk_size
        sub_x_start = 0
    else:                                           # interior tile
        length_x = chunk_size - overlap // 2
        sub_x_start = 0 if col_off == 0 else overlap // 2

    # Height of the valid area
    if offset_y + chunk_size == height:       # touches bottom edge
        length_y = chunk_size
        sub_y_start = 0
    else:                                           # interior tile
        length_y = chunk_size - overlap // 2
        sub_y_start = 0 if row_off == 0 else overlap // 2

    return offset_x, offset_y, length_x, length_y, sub_x_start, sub_y_start
