import cv2 as cv

img = cv.imread('pictures/mancity_team.jpg')
cv.imshow('messi', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray messi', gray)

haar_cascade = cv.CascadeClassifier('haar_detect.xml')

'''trial by smoothing for group image'''

# average = cv.blur(src=gray, ksize=(5,5))
# cv.imshow('Average Blur', average)

gauss = cv.GaussianBlur(src=gray, ksize=(7,7), sigmaX=0)
cv.imshow('Gauss', gauss)

faces_rect = haar_cascade.detectMultiScale(gauss, scaleFactor=1.1, minNeighbors=6)

print(f'Number of faces fpund = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), thickness=2)


cv.imshow('detected', img)
cv.waitKey(0)