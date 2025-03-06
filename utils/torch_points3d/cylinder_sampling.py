class CylinderSampling:
    """ Samples points within a cylinder

    Parameters
    ----------
    radius : float
        Radius of the cylinder
    cylinder_centre : torch.Tensor or np.array
        Centre of the cylinder (1D array that contains (x,y,z) or (x,y))
    align_origin : bool, optional
        move resulting point cloud to origin
    """

    KDTREE_KEY = KDTREE_KEY

    def __init__(self, radius, cylinder_centre, align_origin=True):
        self._radius = radius
        if cylinder_centre.shape[0] == 3:
            cylinder_centre = cylinder_centre[:-1]
        self._centre = np.asarray(cylinder_centre)
        if len(self._centre.shape) == 1:
            self._centre = np.expand_dims(self._centre, 0)
        self._align_origin = align_origin

    def __call__(self, data):
        num_points = data.pos.shape[0]
        if not hasattr(data, self.KDTREE_KEY):
            tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=50)
            setattr(data, self.KDTREE_KEY, tree)
        else:
            tree = getattr(data, self.KDTREE_KEY)

        t_center = torch.FloatTensor(self._centre)
        ind = torch.LongTensor(tree.query_radius(self._centre, r=self._radius)[0])

        new_data = Data()
        for key in set(data.keys):
            if key == self.KDTREE_KEY:
                continue
            item = data[key]
            if torch.is_tensor(item) and num_points == item.shape[0]:
                item = item[ind]
                if self._align_origin and key == "pos":  # Center the cylinder.
                    item[:, :-1] -= t_center
            elif torch.is_tensor(item):
                item = item.clone()
            setattr(new_data, key, item)
        return new_data

    def __repr__(self):
        return "{}(radius={}, center={}, align_origin={})".format(
            self.__class__.__name__, self._radius, self._centre, self._align_origin
        )
