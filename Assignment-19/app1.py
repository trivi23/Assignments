import cv2

img = cv2.imread("happy.png")
size=(225,225)
img = cv2.resize(img,size)
cv2.imwrite('happy.png',img)