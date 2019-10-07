from scipy.spatial import ConvexHull
from matplotlib.path import Path
import matplotlib.pyplot as plt

class ConvexHullHelper(object):

    def __init__(self, points, # numpy.array of points, shape [N,D]
                 ):
        self._points = points
        self._hull = ConvexHull(points)
        self._hull_path = Path(points[self._hull.vertices])

    def plot(self):
        """
        Plot the convex hull
        :return:
        :rtype:
        """
        points = self._points
        hull = self._hull
        plt.plot(points[:, 0], points[:, 1], 'o')
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        plt.show()

    def contains_point(self, pt):
        """
        Checks whether point is contained in the convex hull
        :param pt:
        :type pt:
        :return:
        :rtype:
        """
        return self._hull_path.contains_point(pt)
