import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
import random


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


def elastic_distortion(xyz, granularity, magnitude, flow):
    """Apply elastic distortion on sparse coordinate space.
      pointcloud: numpy array of (number of points, at least 3 spatial dims)
      granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
      magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
    blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
    blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
    xyz_min = xyz.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((xyz - xyz_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(xyz_min - granularity, xyz_min + granularity *
                                   (noise_dim - 2), noise_dim)
    ]
    interp = RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
    xyz += interp(xyz) * magnitude
    flow -= interp(xyz) * magnitude
    return xyz, flow


class ElasticDistortion:

    def __init__(self, distortion_params):
        self.distortion_params = distortion_params

    def __call__(self, xyz, flow):
        if self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    xyz, flow = elastic_distortion(xyz, granularity, magnitude, flow)
        return xyz, flow


class RandomHorizontalFlip(object):

    def __init__(self, upright_axis, is_temporal=False):
        """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords, flow):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coords_after_flow = coords + flow
                    coord_max = np.max(coords[:, curr_ax])
                    coords[:, curr_ax] = coord_max - coords[:, curr_ax]
                    flow[:, curr_ax] = coords_after_flow[:, curr_ax] - coords[:, curr_ax]
        return coords, flow