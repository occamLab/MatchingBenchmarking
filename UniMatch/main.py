from Benchmarcer import Benchmarcer
from Algorithm import Orb, Sift, SuperGlue
import cv2

img1 = cv2.imread("/Users/anmolsandhu/a/cvprac/occam_/othercomp.jpg", 0)    # read b/w image
img2 = cv2.imread("/Users/anmolsandhu/a/cvprac/occam_/othercomp2.jpg", 0)   # read b/w image

new_orb_matcher = Orb(img1, img2)
new_sift_matcher = Sift(img1, img2)
print(new_orb_matcher.get_uni_matches())
print(new_sift_matcher.get_uni_matches())
