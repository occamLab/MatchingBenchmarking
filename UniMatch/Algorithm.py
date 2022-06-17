from abc import ABC, abstractmethod
import cv2
import matplotlib.pyplot as plt
from UniMatch import UNIMatch

class AlgorithmObj(ABC):
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
    
    @abstractmethod
    def draw_matches(self):
        pass

    @abstractmethod
    def get_uni_matches(self):
        pass

class SuperGlue(AlgorithmObj):
    def __init__(self):
        pass

class Orb2(AlgorithmObj):

    def __init__(self, img1, img2):
        super().__init__(img1, img2)
        self.matches, self.kp1, self.kp2 = self.get_matches()
    
    def get_matches(self):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(self.img1, None)
        kp2, des2 = orb.detectAndCompute(self.img2, None)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance), kp1, kp2

    def draw_matches(self):
        # draw first 10 matches
        img3 = cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, self.matches[:50], self.img2, flags=2)
        plt.imshow(img3), plt.show()
    
    def get_uni_matches(self):
        coord_array = []
        for match in self.matches:
            x1, y1 = self.kp1[match.queryIdx].pt
            x2, y2 = self.kp2[match.trainIdx].pt
            for x in [x1, y1, x2, y2]:
                coord_array.append(x)
        # create list of UNIMatch objects
        UNIlist = []
        for i in range(0, len(coord_array), 4):
            UNIlist.append(UNIMatch(coord_array[i], coord_array[i+1], coord_array[i+2], coord_array[i+3]))
        
        return UNIlist

class Sift(AlgorithmObj):
    def __init__(self, img1, img2):
        super().__init__(img1, img2)
        self.good_matches, self.kp1, self.kp2 = self.get_matches()

    def get_matches(self):
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.img1, None)
        kp2, des2 = sift.detectAndCompute(self.img2, None)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test (currently set to 1 so it does nothing)
        good_matches = [m for m, n in matches if m.distance < 1 * n.distance]
        # for m, n in matches:
        #     if m.distance < 1*n.distance:
        #         good_matches.append(m)
        return good_matches, kp1, kp2

    def draw_matches(self):
        # draw first 10 matches
        img3 = cv2.drawMatchesKnn(self.img1, self.kp1, self.img2, self.kp2, self.good_matches, None, flags=2)
        plt.imshow(img3), plt.show()
    
    def get_uni_matches(self):
        coord_array = []
        for match in self.good_matches:
            x1, y1 = self.kp1[match.queryIdx].pt
            x2, y2 = self.kp2[match.trainIdx].pt
            for x in [x1, y1, x2, y2]:
                coord_array.append(x) 
        # create list of UNIMatch objects
        UNIlist = []
        for i in range(0, len(coord_array), 4):
            UNIlist.append(UNIMatch(coord_array[i], coord_array[i+1], coord_array[i+2], coord_array[i+3]))

        return UNIlist