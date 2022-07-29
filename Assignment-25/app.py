import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
filename='image.jpg'
img=cv2.imread(filename)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
fl=face.detectMultiScale(gray,1.09,7)
monkey=cv2.imread('monkey.jpg')


def put_monkey_filter(monkey, fc, x, y, w, h):
    face_width = w
    face_height = h

    monkey = cv2.resize(monkey, (int(face_width * 1.5), int(face_height * 1.95)))
    for i in range(int(face_height * 1.75)):
        for j in range(int(face_width * 1.5)):
            for k in range(3):
                if monkey[i][j][k] < 235:
                    fc[y + i - int(0.375 * h) - 1][x + j - int(0.35 * w)][k] = monkey[i][j][k]
    return fc

for (x, y, w, h) in fl:
    frame = put_monkey_filter(monkey, img, x, y, w, h)
    cv2.imshow('image',frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break