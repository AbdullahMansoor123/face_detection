import cv2
import numpy as np


def face_detection(frame, net, scale, size, mean):
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=size, mean=mean, swapRB=False, crop=False)
    net.setInput(blob)
    detection = net.forward()
    w = frame.shape[1]
    h = frame.shape[0]

    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]

        if confidence > detection_threshold:
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')

            # Annotate the video frame with the detection results.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = 'Confidence: %.4f' % confidence

            label_size, base_line = cv2.getTextSize(label, font_style, font_scale, font_thickness)
            cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x1, y1), font_style, font_scale, (0, 0, 0))
    return frame


# Annotation settings.
font_style = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1

scale = 1.0
size = (300, 300)
mean = [104, 117, 123]
detection_threshold = 0.4

net = cv2.dnn.readNetFromCaffe('./model/deploy.prototxt',
                               './model/res10_300x300_ssd_iter_140000.caffemodel')

# Detection on Image
# image = cv2.imread('team.png')
# frame = face_detection(image, net, scale, size, mean)
# cv2.imshow('image', frame)
# if cv2.waitKey(0) == ord('q'):
#     print('')


# Detection on Video
vcap = cv2.VideoCapture(0)
while True:
    ret, frame = vcap.read()
    if not ret:
        break

    frame = face_detection(frame, net, scale, size, mean)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
#
vcap.release()
cv2.destroyAllWindows()
