from functools import partial

import matplotlib.colors as mcolors
import numpy as np

# user-defined color maps
discrete_dict = {
    "defne": ["#03BDAB", "#FEAC16", "#5D00E6", "#F2BCF3", "#AFEEEE", "#E92C91"],
    "aqua": [
        "#f7fcf0",
        "#ccebc5",
        "#7bccc4",
        "#4eb3d3",
        "#2b8cbe",
        "#08589e",
        "#1b3f87",
        "#18347a",
    ],
}
continuous_dict = {
    "defne": ["#03BDAB", "#FEAC16", "#5D00E6"],
    "aqua": [
        "#f7fcf0",
        "#ccebc5",
        "#7bccc4",
        "#4eb3d3",
        "#2b8cbe",
        "#08589e",
        "#1b3f87",
        "#18347a",
    ],
}


def create_discrete_colormap(colors, name="custom_colormap"):
    """Create a discrete colormap from given color hex codes.

    Args:
        colors (list): List of color hex codes.
        name (str, optional): Name of the colormap. Defaults to 'custom_colormap'.

    Returns:
        matplotlib.colors.ListedColormap: The discrete colormap object.
    """
    cmap = mcolors.ListedColormap(colors, name=name)
    return cmap


def create_continuous_colormap(colors, name="custom_colormap", N=256):
    """Create a continuous colormap from given color hex codes.

    Args:
        colors (list): List of color hex codes.
        name (str, optional): Name of the colormap. Defaults to 'custom_colormap'.
        N (int, optional): Number of color levels. Defaults to 256.

    Returns:
        matplotlib.colors.ListedColormap: The continuous colormap object.
    """
    ncolors = len(colors)
    if ncolors < 2:
        raise ValueError("Please provide at least two colors.")

    color_array = np.zeros((N, 4))
    for i in range(N):
        idx1 = int(i * (ncolors - 1) / N)
        idx2 = min(idx1 + 1, ncolors - 1)
        t = i * (ncolors - 1) / N - idx1
        color_array[i] = tuple(
            (1 - t) * c1 + t * c2
            for c1, c2 in zip(
                mcolors.to_rgba(colors[idx1]), mcolors.to_rgba(colors[idx2])
            )
        )
    cmap = mcolors.ListedColormap(color_array, name=name)
    return cmap


def create_custom_colormap(map_name="defne", type="discrete", colors=None, N=256):
    """Create a custom colormap.

    This function creates either a discrete or continuous colormap based on the given parameters.

    Args:
        map_name (str, optional): Name of the custom colormap. Defaults to 'defne'.
        cmap_type (str, optional): Type of the colormap ('discrete' or 'continuous'). Defaults to 'discrete'.
        colors (list, optional): List of color hex codes. If None, uses predefined colormap based on map_name. Defaults to None.
        N (int, optional): Number of color levels for continuous colormap. Defaults to 256.

    Returns:
        matplotlib.colors.ListedColormap: The custom colormap object.
    """
    colors_dict = {"discrete": discrete_dict, "continuous": continuous_dict}
    function_dict = {
        "discrete": create_discrete_colormap,
        "continuous": partial(create_continuous_colormap, N=N),
    }
    if colors:
        assert isinstance(colors, list)
        colors_hex = colors
    else:
        try:
            colors_hex = colors_dict[type][map_name]
        except KeyError:
            print(f"map {map_name} does not exist.")
            raise NotImplementedError
    cmap_fn = function_dict[type]
    cmap = cmap_fn(colors_hex, map_name)
    return cmap
