import numpy as np
import cv2 as cv

# default settings

CAMERA_ID = 1
HEIGHT = 1080
WIDTH = 1920
FPS = 30


def test_video_capture() -> None:
    capture = cv.VideoCapture(CAMERA_ID) #TODO try catch here
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv.CAP_PROP_FPS, FPS)
    while True:
        is_true, frame = capture.read()
        cv.imshow("Video Capture Test", frame)

        if cv.waitKey(20) & 0xFF == ord('c'):
            break

    print(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    print(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(capture.get(cv.CAP_PROP_FPS))
    # print(capture.get(cv.CAP_PROP_BACKEND))
    capture.release()
    cv.destroyAllWindows()


test_video_capture()
