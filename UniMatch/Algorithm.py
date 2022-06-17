from abc import ABC, abstractmethod
import cv2
import matplotlib.pyplot as plt

class AlgorithmObj(ABC):
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
    
    @abstractmethod
    def draw_matches(self):
        pass

    @abstractmethod
    def coord_array(self):
        pass

class SuperGlue(AlgorithmObj):
    def __init__(self):
        pass

class Orb(AlgorithmObj):
    def __init__(self, img1, img2):
        super().__init__(img1, img2)
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with SIFT
        self.kp1, self.des1 = orb.detectAndCompute(self.img1, None)
        self.kp2, self.des2 = orb.detectAndCompute(self.img2, None)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.matches = bf.match(self.des1, self.des2)
        # sort matches by distance between descriptors
        self.matches = sorted(self.matches, key=lambda x: x.distance)
    
    def draw_matches(self):
        # draw first 10 matches
        img3 = cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, self.matches[:50], self.img2, flags=2)
        plt.imshow(img3), plt.show()
    
    def coord_array(self):
        coord_array = []
        for match in self.matches:
            x1, y1 = self.kp1[match.queryIdx].pt
            x2, y2 = self.kp2[match.trainIdx].pt
            coord_array.append(x1)
            coord_array.append(y1)
            coord_array.append(x2)
            coord_array.append(y2)
        return coord_array

class Sift(AlgorithmObj):
    def __init__(self, img1, img2):
        super().__init__(img1, img2)
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        self.kp1, self.des1 = sift.detectAndCompute(self.img1, None)
        self.kp2, self.des2 = sift.detectAndCompute(self.img2, None)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        self.matches = bf.knnMatch(self.des1, self.des2, k=2)
        # sort matches by distance between descriptors (doesn't work on knnMatch)
        # self.matches = sorted(self.matches, key=lambda x: x.distance)
        # Apply ratio test
        self.good_matches = []
        for m, n in self.matches:
            if m.distance < 0.4*n.distance:
                self.good_matches.append([m])                

    def draw_matches(self):
        # draw first 10 matches
        img3 = cv2.drawMatchesKnn(self.img1, self.kp1, self.img2, self.kp2, self.good_matches, None, flags=2)
        plt.imshow(img3), plt.show()
    
    def coord_array(self):
        coord_array = []
        for m,n in self.good_matches:
            x1, y1 = self.kp1[m[0].queryIdx].pt
            x2, y2 = self.kp2[n[0].trainIdx].pt
            for x in [x1, y1, x2, y2]:
                coord_array.append(x) 
        return coord_array