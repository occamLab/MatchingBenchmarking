from Benchmarcer import Benchmarcer
from Algorithm import Orb, Sift, SuperGlue
import cv2

img1 = cv2.imread("test.jpg", 0)    # read b/w image
img2 = cv2.imread("test2.jpg", 0)   # read b/w image

new_orb_matcher = Orb(img1, img2)
new_sift_matcher = Sift(img1, img2)
print(new_orb_matcher.coord_array())
print(new_sift_matcher.coord_array())
