class UNIPoint:
    """
    A data structure representing a keypoint in an image.
    Analogous to cv2.KeyPoint.

    Instance Attributes:
        x (float): The x coordinate of the keypoint.
        y (float): The y coordinate of the keypoint.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"UNIPoint({self.x}, {self.y})"


class UNIMatch:
    """
    A data structure representing matched UNIPoints between two images.
    This object was created to have one uniform match object since different
    image matching algorithms and models use different data structures.

    Instance Attributes:
        queryPt (UNIPoint): The UNIPoint in the query image.
        trainPt (UNIPoint): The UNIPoint in the train image.
    """

    def __init__(self, x1, y1, x2, y2):
        self.queryPt = UNIPoint(x1, y1)
        self.trainPt = UNIPoint(x2, y2)

    def __repr__(self):
        return f"UNIMatch({self.queryPt}, {self.trainPt})"
