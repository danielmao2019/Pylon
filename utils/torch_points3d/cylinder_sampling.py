import numpy as np
import torch
from sklearn.neighbors import KDTree


class CylinderSampling:
    """ Samples points within a cylinder

    Parameters
    ----------
    radius : float
        Radius of the cylinder
    cylinder_centre : torch.Tensor or np.array
        Centre of the cylinder (1D array that contains (x,y,z))
    align_origin : bool, optional
        move resulting point cloud to origin
    """

    def __init__(self, radius, cylinder_centre, align_origin=False):
        self._radius = radius
        self._centre = np.asarray(cylinder_centre)
        if len(self._centre.shape) == 1:
            self._centre = np.expand_dims(self._centre, 0)
        self._align_origin = align_origin

    def query(self, kdtree: KDTree, points: torch.Tensor) -> torch.Tensor:
        """Query points within the cylinder using a KDTree

        Parameters
        ----------
        kdtree : KDTree
            KDTree built from points
        points : torch.Tensor
            Points to sample from

        Returns
        -------
        torch.Tensor
            Indices of points within the cylinder
        """
        # Get indices of points within radius
        indices = torch.LongTensor(kdtree.query_radius(self._centre, r=self._radius)[0])
        
        if self._align_origin and len(indices) > 0:
            points[indices] = points[indices] - torch.FloatTensor(self._centre)
            
        return indices

    def __repr__(self):
        return "{}(radius={}, center={}, align_origin={})".format(
            self.__class__.__name__, self._radius, self._centre, self._align_origin
        )
