import numpy as np
import cv2 as cv
import tkinter

# default settings

CAMERA_ID = 1
HEIGHT = 1080
WIDTH = 1920
FPS = 30


def test_video_capture() -> None:
    capture = cv.VideoCapture(CAMERA_ID)  # TODO try catch here
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv.CAP_PROP_FPS, FPS)
    while True:
        is_true, frame = capture.read()
        cv.imshow("Video Capture Test", frame)

        key_pressed = cv.waitKey(20)
        if key_pressed == ord('c'):
            break
        elif key_pressed == ord('s'):
            cv.imwrite("Training Data/test.jpg", frame)


    print(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    print(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(capture.get(cv.CAP_PROP_FPS))
    # print(capture.get(cv.CAP_PROP_BACKEND))
    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    test_video_capture()

