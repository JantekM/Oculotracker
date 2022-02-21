import numpy as np
import cv2 as cv
from PIL import Image, ImageTk

# default settings

CAMERA_ID = 1
HEIGHT = 1080
WIDTH = 1920
FPS = 30

capture = None
wait_period: int = 1000
#frame = None


def start_capture(cam = CAMERA_ID, height = HEIGHT, width = WIDTH, fps = FPS):
    print("starting video capture")
    global capture, wait_period
    capture = cv.VideoCapture(cam)

    #capture.release()
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv.CAP_PROP_FPS, fps)

    working, _ = capture.read()
    if not working:
        # TODO log camera error
        print("can't open camera")
        capture = None
        return False

    wait_period = int (1000/fps)
    return True


def stop_capture():
    global capture
    capture.release()

def get_frame_cv():
    global capture
    working, frame = capture.read()
    if not working:
        raise BaseException('error reading from camera input')
    return frame

def get_frame_tk(canvas_height: int, canvas_width: int):
    global capture
    frame = get_frame_cv()
    capture_height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    capture_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)

    scale = min(canvas_height / capture_height, canvas_width / capture_width)

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)

    resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)



    #cv.imshow('first', resized)
    blue, green, red = cv.split(resized)
    img = cv.merge((red, green, blue))
    #cv.imshow('second', img)
    im = Image.fromarray(img)
    return ImageTk.PhotoImage(image=im)


def tst_video_capture() -> None:
    start_capture()

    frame = get_frame_cv()

    while True:
        frame = get_frame_cv()
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
    tst_video_capture()

