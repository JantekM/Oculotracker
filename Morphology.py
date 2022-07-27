import cv2
import numpy as np

def blob_process(img, detector):
    img = cv2.erode(img, None, iterations=2) #1
    img = cv2.dilate(img, None, iterations=4) #2
    img = cv2.medianBlur(img, 5) #3
    blob = detector.detect(img)
    assert len(blob) > 0
    if len(blob) > 1:
        blob_max = blob[0]
        size_max = blob[0].size
        for blb in blob:
            if blb.size > size_max:
                blob_max = blb
        blob = blob_max
    else:
        blob = blob[0]
    return {"coords": blob.pt, "size": blob.size}

def tophat(img):
    tophat_size = 10  # TODO parametr
    tophat_shape = cv2.MORPH_ELLIPSE
    operation = cv2.MORPH_TOPHAT

    element = cv2.getStructuringElement(tophat_shape, (2 * tophat_size + 1, 2 * tophat_size + 1),
                                        (tophat_size, tophat_size))

    tophat_dst = cv2.morphologyEx(img, operation, element)
    return tophat_dst


def gradient(blurred):
    scale: float = 2  # TODO: parametr do modelu
    delta: float = 0
    ddepth = cv2.CV_16S
    # grad_x = cv2.Scharr(blurred,ddepth,1,0)
    grad_x = cv2.Sobel(blurred, ddepth, 1, 0, ksize=1, scale=scale, delta=delta,
                       borderType=cv2.BORDER_DEFAULT)  # TODO: parametr do modelu
    # Gradient-Y
    # grad_y = cv2.Scharr(blurred,ddepth,0,1)
    grad_y = cv2.Sobel(blurred, ddepth, 0, 1, ksize=1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def connected_segments(img, n=2):
    num_seg, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    ind = stats[:, cv2.CC_STAT_AREA].argsort()[range(num_seg - n - 1, num_seg - 1)]
    ind_sorted = ind[stats[ind, cv2.CC_STAT_LEFT].argsort()]
    return {"stats": stats[ind_sorted, :], "centroids": centroids[ind_sorted, :]}


def analyze_ROI(image, offset, debug=False):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)

    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByCircularity
    detector_params.minCircularity = 0.35
    detector = cv2.SimpleBlobDetector_create(detector_params)

    blurred = cv2.blur(img, (3, 3))  # TODO: parametr do modelu
    tophatted = tophat(img)
    grad = gradient(blurred)

    # TODO: parametry do modelu
    # 24 sama źrenica
    # 32 zarys powieki
    # 62 zarys kątów oka

    _, thr1 = cv2.threshold(blurred, 24, 255, cv2.THRESH_BINARY)
    _, thr2 = cv2.threshold(blurred, 32, 255, cv2.THRESH_BINARY)
    _, thr3 = cv2.threshold(blurred, 62, 255, cv2.THRESH_BINARY)
    _, thr4 = cv2.threshold(tophatted, 75, 255, cv2.THRESH_BINARY)

    white_triangles = connected_segments(thr4, n=2)
    iris_and_reflex = connected_segments(thr1, n=2)
    eye_borders = connected_segments(thr3, n=1)
    eye_borders_v2 = connected_segments(thr2, n=1)
    blob1 = blob_process(thr1, detector)
    blob2 = blob_process(thr2, detector)
    blob3 = blob_process(thr3, detector)
    blobs = blob1, blob2, blob3
    if debug:
        cv2.imshow("equalized", img)
        # cv2.imshow("blurred", blurred)
        # cv2.imshow("grad", grad)
        cv2.imshow("thr1", thr1)
        cv2.imshow("thr2", thr2)
        cv2.imshow("thr3", thr3)
        cv2.imshow("thr4", thr4)
        cv2.imshow("tophatted", tophatted)
        cv2.waitKey()
        cv2.destroyAllWindows()
        print(white_triangles, iris_and_reflex, eye_borders, eye_borders_v2)

    return white_triangles, iris_and_reflex, eye_borders, eye_borders_v2, blobs


def analyze_eyes(ROIs, debug=False):
    ROI_right, ROI_left, ROI_coords_right, ROI_coords_left = ROIs
    landmarks_right = analyze_ROI(ROI_right, (ROI_coords_right[0], ROI_coords_right[2]), debug)
    landmarks_left = analyze_ROI(ROI_left, (ROI_coords_left[0], ROI_coords_left[2]), debug)
    if debug:
        print(landmarks_right, landmarks_left, ROI_coords_right, ROI_coords_left)
    return landmarks_right, landmarks_left, ROI_coords_right, ROI_coords_left
