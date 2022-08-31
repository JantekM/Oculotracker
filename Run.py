import tkinter
from queue import Queue

import Modelling
import PrepareDataset
import video_capture
import cv2 as cv
import tensorflow as tf
import numpy as np
import win32api, win32con


root = None
model = tf.keras.models.load_model('Models/main', custom_objects={'custom_loss': Modelling.custom_loss})

last_n_coords = 5

xs = [0, 0, 0, 0, 0]
ys = [0, 0, 0, 0, 0]

def key_handler(event):
    # Replace the window's title with event.type: input key
    #root.title("{}: {}".format(str(event.type), event.keysym))
    ctrl_pressed = (event.state & 0x4) != 0
    if event.keysym == 'Escape':
        video_capture.stop_capture()
        cv.destroyAllWindows()
        root.destroy()



def key_release_handler(event):
    if event.keysym == 'space':
        pass


def render():
    global root
    root.update()
    root.img = video_capture.get_frame_cv()
    input = PrepareDataset.prepare_single_frame(root.img)
    #np.save('input.npz', input)
    #input = input.reshape(1, 1544)
    if input is not None:
        try:
            coords = model.predict(input.reshape(1, 1544), verbose=0)
            #print(coords)
            x, y = coords[0][0], coords[0][1]
            xs.append(x)
            ys.append(y)
            xs.pop(0)
            ys.pop(0)

            win32api.SetCursorPos((np.mean(xs), np.mean(ys)))
        except Exception:
            print("other error")

    root.after(video_capture.wait_period, render)


def main() -> None:
    global root
    root = tkinter.Tk()
    root.attributes('-fullscreen', True)
    root.attributes('-alpha', 0.1)
    root.config(cursor="target")
    #root.overrideredirect(True)

    # root.title("Oculotracker Training Data Acquisition Tool")
    root.bind('<Key>', key_handler)
    root.bind('<KeyRelease>', key_release_handler)
    video_capture.start_capture()
    root.after(500, render)

    root.mainloop()
    video_capture.stop_capture()


if __name__ == "__main__":
    main()
