import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe('./model/deploy.prototxt',
                               './model/res10_300x300_ssd_iter_140000.caffemodel')


detection_threshold = 0.5
# Annotation settings.
font_style = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1

vcap = cv2.VideoCapture(0)
while True:
    ret, frame = vcap.read()
    if not ret:
        break

    scale_factor = 0.1
    size = (300, 300)
    w = frame.shape[0]
    h = frame.shape[1]
    mean = []
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale_factor, size=size, mean=mean, swapRB=False, crop=False)
    net.setInput(blob)
    detection = net.forward()

    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > detection_threshold:
            box = detection[0,0,i,3:7] * np.array([w,h,w,h])
            (x1, y1, x2, y2) = box.astype('int')

            # Annotate the video frame with the detection results.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = 'Confidence: %.4f' % confidence
            label_size, base_line = cv2.getTextSize(label, font_style, font_scale, font_thickness)
            cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x1, y1), font_style, font_scale, (0, 0, 0))

            cv2.imshow('frame', frame)
