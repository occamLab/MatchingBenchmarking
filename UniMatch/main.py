from Benchmarcer import Benchmarcer
from Algorithm import Orb, Sift, SuperGlue
import cv2

img1 = cv2.imread("/Users/marce/Desktop/MatchingBenchmarking/UniMatch/othercomp.jpg")    # read b/w image
img2 = cv2.imread("/Users/marce/Desktop/MatchingBenchmarking/UniMatch/othercomp2.jpg")   # read b/w image

# cv2.imshow("img1", img1)
new_orb_matcher = Orb(img1, img2)
new_sift_matcher = Sift(img1, img2)
# print(new_orb_matcher.get_uni_matches())
print(new_sift_matcher.get_uni_matches())
