class UNIPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"UNIPoint({self.x}, {self.y})"

class UNIMatch:
    def __init__(self, x1, y1, x2, y2):
        self.queryPt = UNIPoint(x1, y1)
        self.trainPt = UNIPoint(x2, y2)

    def __repr__(self):
        return f"UNIMatch({self.queryPt}, {self.trainPt})"
