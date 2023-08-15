import cv2
import numpy as np

config_file = './model/deploy.prototxt'
model_file = './model/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(config_file, model_file)


def blur(face, factor=3):
    h, w = face.shape[:2]

    if factor < 1:
        factor = 1
    if factor > 5:
        factor = 5

    w_k = int(w / factor)
    h_k = int(h / factor)

    if w_k % 2 == 0: w_k += 1
    if h_k % 2 == 0: h_k += 1
    blurred = cv2.GaussianBlur(face, (int(w_k), int(h_k)), 0, 0)
    return blurred


def face_blur_ellipse(frame, net, factor):
    img = frame.copy()
    img_out = frame.copy()

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
            (x1, y1, x2, y2) = box

            # extract face roi
            face_roi = img[int(y1):int(y2), int(x1):int(x2)]

            blur_face = blur(face_roi, factor=factor)
            blur_pixel_face = roi_pixelated(blur_face)

            # replace detection face with blur one
            img_out[int(y1):int(y2), int(x1):int(x2)] = blur_pixel_face

            # Specify the elliptical parameters directly from the bounding box coordinates.
            e_center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
            e_size = (x2 - x1, y2 - y1)
            e_angle = 0.0

            # Draw a ellipse with red line borders of thickness of 5 px
            elliptical_mask = cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle), (255, 255, 255), -1)

            np.putmask(img, elliptical_mask, img_out)
        else:
            pass
    return img


def roi_pixelated(face_roi, pixels=20):
    roi_h, roi_w = face_roi.shape[0], face_roi.shape[1]
    if roi_h > pixels and roi_w > pixels:
        roi_small = cv2.resize(face_roi, (pixels, pixels), interpolation=cv2.INTER_LINEAR)
        pixlated_roi = cv2.resize(roi_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        return pixlated_roi


def face_pixelated(frame):
    img = frame.copy()

    blob = cv2.dnn.blobFromImage(img, scalefactor=scale, size=size, mean=mean)
    net.setInput(blob)
    detection = net.forward()
    h, w = img.shape[:2]

    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > detection_threshold:
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')

            # extract face roi
            face_roi = img[y1:y2, x1:x2]
            face_roi = roi_pixelated(face_roi)
            # replace detection face with blur one
            img[y1:y2, x1:x2] = face_roi
            # print(img)
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
    # frame = face_pixelated(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
#
vcap.release()
cv2.destroyAllWindows()
