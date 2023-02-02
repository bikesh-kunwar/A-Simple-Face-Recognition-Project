import cv2
import io
camera_port = 0
#cam = cv2.VideoCapture(camera_port)
cam= cv2.VideoCapture(camera_port,cv2.CAP_DSHOW)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

Id = input('enter your id')
sampleNum = 1
while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # incrementing sample number
        sampleNum = sampleNum + 1
        # saving the captured face in the dataset folder
        cv2.imwrite("dataset/User." + str(Id) + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('frame', img)
    # wait for 100 miliseconds
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum > 20:
        break
cam.release()
cv2.destroyAllWindows()
