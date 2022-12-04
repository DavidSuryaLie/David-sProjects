import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('data/hilux.jpg')
img_scene = cv2.imread('data/hiluxrotated.jpg')

SIFT = cv2.SIFT_create()

kp, dsc = SIFT.detectAndCompute(img, None)
kp_scene, desc_scene = SIFT.detectAndCompute(img_scene, None)

img_kp = cv2.drawKeypoints(img, kp, None)
plt.figure(figsize=(10, 6))
plt.imshow(img_kp, cmap="gray")
plt.title("img_keypoints")
plt.axis('off')
plt.show()
