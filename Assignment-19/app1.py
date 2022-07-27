import cv2

img = cv2.imread("grey.jpg")
size=(225,225)
img = cv2.resize(img,size)
cv2.imwrite('grey.jpg',img)