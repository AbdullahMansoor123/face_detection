import cv2
import numpy as np

config_file = './model/deploy.prototxt'
model_file = './model/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(config_file, model_file)


def blur(face, factor=3):
    h, w = face.shape[:2]

    if factor < 1: factor = 1
    if factor > 5: factor = 5

    w_k = int(w / factor)
    h_k = int(h / factor)

    if w_k % 2 == 0: w_k += 1
    if h_k % 2 == 0: h_k += 1
    blurred = cv2.GaussianBlur(face, (int(w_k), int(h_k)), 0, 0)
    return blurred


def face_blur_ellipse(frame, net, factor):
    img = frame.copy()
    img_blur = frame.copy()
    elliptical_mask = np.zeros(frame.shape, dtype=img.dtype)

    blob = cv2.dnn.blobFromImage(img, scalefactor=scale, size=size, mean=mean)
    net.setInput(blob)
    detection = net.forward()
    w = img.shape[1]
    h = img.shape[0]

    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]

        if confidence > detection_threshold:
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')

            # extract face roi
            face_roi = img[y1:y2, x1:x2]
            blur_face = blur(face_roi, factor=factor)

            # replace detection face with blur one
            img_blur[y1:y2, x1:x2] = blur_face

            # Specify the elliptical parameters directly from the bounding box coordinates.
            e_center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
            print(e_center)
            e_size = (x2 - x1, y2 - y1)
            print(e_size)
            e_angle = 0.0
            ellipse_float = (e_center, e_size, e_angle)
            # elliptical_mask = cv2.ellipse(elliptical_mask, , color=(255, 255, 255),
            # thickness=-1,
            #                               lineType=cv2.LINE_AA)
            # elliptical_mask=cv2.ellipse(elliptical_mask, e_center, e_size, e_angle,)
            np.putmask(img, elliptical_mask, img_blur)

    return img


# model parameter
scale = 1.0
size = (300, 300)
mean = [104, 117, 123]
detection_threshold = 0.9

# # Detection on Image
# image = cv2.imread('team.png')
# frame = face_blur(image, net, factor=3)
# cv2.imshow('image', frame)
# if cv2.waitKey(0) == ord('q'):
#     print('')


vcap = cv2.VideoCapture(0)
while True:
    ret, frame = vcap.read()
    if not ret:
        break

    frame = face_blur_ellipse(frame, net, factor=3)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
#
vcap.release()
cv2.destroyAllWindows()
