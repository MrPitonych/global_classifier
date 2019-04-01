import numpy as np


def projections(x, y, z, mode=1):
    """ Function for calculate find vertical and horizontal projections.

                                Args:
                                    x (list): Series data from x channel.
                                    y (list): Series data from y channel.
                                    z (list): Series data from z channel.
                                    mode (int): Operation mode: 1 = find horizontal projection,
                                                                2 = find vertical projections.
                                                                Default 1.
                                Returns:
                                    list: The return value correlation. Size: time series/timewindow.
        """
    acc = np.stack([x, y, z])
    gravity = np.mean(acc, axis=1).reshape(len(acc), 1)
    acc = acc - gravity
    av = ((sum(gravity * acc) / (sum(gravity * gravity))) * gravity)
    h = acc - av
    horizontal = np.sqrt(np.sum((h ** 2), axis=0))

    if mode == 1:
        return horizontal
    elif mode == 2:
        return av
