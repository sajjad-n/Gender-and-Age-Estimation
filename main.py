import cv2
import numpy
from wide_resnet import WideResNet

# select input method
webcam = True

# initialize model
model = WideResNet()()
model.load_weights('model/weights.18-4.06.hdf5') #https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5

# initialize webcam
video_capture = cv2.VideoCapture(0)

# initialize face detection classifier
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_alt.xml')

while True:
    if webcam:
        _, frame = video_capture.read()
    else:
        frame = cv2.imread('test/photo.jpg')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        print('Found', len(faces), 'faces.')
    else:
        print('looking for faces...')

    for i in range(0, len(faces)):
        # Get Face
        (x, y, w, h) = faces[i]
        cropped_face = frame[int(y):int(y + h), int(x):int(x + w)]
        detected_face = cv2.resize(cropped_face, (64, 64))

        # copy detected face in an array of 4 dimensions
        face_imgs = numpy.empty((len(faces), 64, 64, 3))
        face_imgs[i, :, :, :] = detected_face

        # predict
        result = model.predict(face_imgs)
        predicted_genders = result[0]
        ages = numpy.arange(0, 101).reshape(101, 1)
        predicted_ages = result[1].dot(ages).flatten()

        # set label
        label = ''
        label += 'F' if predicted_genders[i][0] > 0.5 else 'M'
        label += ' ' + str(int(predicted_ages[i]))
        if predicted_ages[i] > 18:
            label += ' adult'
        else:
            label += ' young'

        # show results
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # show frame
    cv2.imshow('frame', frame)

    # end process
    if webcam:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.waitKeyEx(0)
        break
