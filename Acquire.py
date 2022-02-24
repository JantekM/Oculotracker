import os
import random
import time
import tkinter
import tkinter.ttk
import tkinter.filedialog

import video_capture
import cv2 as cv

root = None


class AcquisitionGui:
    gui = None  # singleton instance of the Gui class

    def __init__(self, master=None):
        if AcquisitionGui.gui is None:
            AcquisitionGui.gui = self
        else:
            raise BaseException('trying to overwrite the singleton instance')

        master.title("Oculotracker Training Data Acquisition Tool")

        self.showPreview = tkinter.BooleanVar(value=True)
        self.path = tkinter.StringVar(value=os.getcwd() + '\\Training Data')
        self.recordingMode = False

        # build ui
        self.Window = tkinter.ttk.Panedwindow(master, orient='vertical')
        self.MainPaneFrame = tkinter.ttk.Frame(self.Window)

        self.OptionsFrame = tkinter.ttk.Frame(self.MainPaneFrame)

        self.CameraSettingsFrame = tkinter.ttk.Labelframe(self.OptionsFrame)

        self.CameraIDLabel = tkinter.ttk.Label(self.CameraSettingsFrame)
        self.CameraIDLabel.configure(text='Camera')
        self.CameraIDLabel.grid(column='0', row='0')
        self.CameraSelection = tkinter.ttk.Combobox(self.CameraSettingsFrame)
        self.CameraSelection.grid(column='1', row='0')

        self.ResolutionLabel = tkinter.ttk.Label(self.CameraSettingsFrame)
        self.ResolutionLabel.configure(text='Resolution')
        self.ResolutionLabel.grid(column='0', row='1')
        self.ResolutionSelection = tkinter.ttk.Combobox(self.CameraSettingsFrame)
        self.ResolutionSelection.grid(column='1', row='1')

        self.FPSLabel = tkinter.ttk.Label(self.CameraSettingsFrame)
        self.FPSLabel.configure(text='FPS')
        self.FPSLabel.grid(column='0', row='2')
        self.FPSSelection = tkinter.ttk.Combobox(self.CameraSettingsFrame)
        self.FPSSelection.grid(column='1', row='2')

        self.ShowPreviewCheckbox = tkinter.ttk.Checkbutton(self.CameraSettingsFrame)
        self.ShowPreviewCheckbox.configure(text='Show preview')
        self.ShowPreviewCheckbox.grid(column='0', columnspan='2', row='3')
        self.ShowPreviewCheckbox.configure(variable=self.showPreview)

        self.CameraSettingsFrame.configure(height='200', text='Camera Settings', width='150')
        self.CameraSettingsFrame.pack(fill='both', ipadx='5', ipady='5', side='top')

        self.SeriesSettingsFrame = tkinter.ttk.Labelframe(self.OptionsFrame)

        self.DataPathText = tkinter.Message(self.SeriesSettingsFrame)
        self.DataPathText.configure(justify='left', takefocus=False,
                                    textvariable=self.path,
                                    width='200')
        self.DataPathText.grid(column='1', row='0')
        self.DataPathLabel = tkinter.ttk.Label(self.SeriesSettingsFrame)
        self.DataPathLabel.configure(text='Data path:')
        self.DataPathLabel.grid(column='0', row='0')

        self.ChangePathButton = tkinter.ttk.Button(self.SeriesSettingsFrame)
        self.ChangePathButton.configure(text='Change path')
        self.ChangePathButton.grid(column='0', columnspan='2', padx='10', pady='10', row='1')
        self.ChangePathButton.configure(command=self.ChangePathCallback)

        self.PersonLabel = tkinter.ttk.Label(self.SeriesSettingsFrame)
        self.PersonLabel.configure(text='Person:')
        self.PersonLabel.grid(column='0', padx='5', row='2')

        self.PersonEntry = tkinter.ttk.Entry(self.SeriesSettingsFrame)
        self.PersonEntry.configure(state='disabled', width='30')
        self.PersonEntry['state'] = 'normal'
        self.PersonEntry.delete('0', 'end')
        self.PersonEntry.insert('0', 'Jantek Mikulski')
        self.PersonEntry['state'] = 'disabled'
        self.PersonEntry.grid(column='1', row='2')

        self.PlaceLabel = tkinter.ttk.Label(self.SeriesSettingsFrame)
        self.PlaceLabel.configure(text='Place:')
        self.PlaceLabel.grid(column='0', padx='5', row='3')
        self.PlaceEntry = tkinter.ttk.Entry(self.SeriesSettingsFrame)
        self.PlaceEntry.configure(width='30')
        self.PlaceEntry.grid(column='1', row='3')

        self.LightLabel = tkinter.ttk.Label(self.SeriesSettingsFrame)
        self.LightLabel.configure(text='Light conditions:')
        self.LightLabel.grid(column='0', padx='5', row='4')
        self.LightEntry = tkinter.ttk.Entry(self.SeriesSettingsFrame)
        self.LightEntry.configure(width='30')
        self.LightEntry.grid(column='1', row='4')

        self.AdditionalInfoLabel = tkinter.ttk.Label(self.SeriesSettingsFrame)
        self.AdditionalInfoLabel.configure(text='Additional info:')
        self.AdditionalInfoLabel.grid(column='0', padx='5', row='5')
        self.AdditionalInfoEntry = tkinter.ttk.Entry(self.SeriesSettingsFrame)
        self.AdditionalInfoEntry.configure(width='30')
        self.AdditionalInfoEntry.grid(column='1', row='5')

        self.SeriesSettingsFrame.configure(height='200', text='Series Settings', width='200')
        self.SeriesSettingsFrame.pack(expand='true', fill='both', ipadx='5', ipady='5', side='top')

        self.KeyBindingsFrame = tkinter.ttk.Labelframe(self.OptionsFrame)
        self.KeyBindingsLabel = tkinter.ttk.Label(self.KeyBindingsFrame)
        self.KeyBindingsLabel.configure(text='''Show keybindings: H
        Exit Fullscreen mode: Esc
        Record: Space
        Launch Constant Pose Mode: Ctrl + P
        Launch Cursor Mode: Ctrl + M
        Record out of border: Ctrl + B 
        ''')
        self.KeyBindingsLabel.pack(side='top')
        self.KeyBindingsFrame.configure(height='200', text='Key bindings:', width='200')
        self.KeyBindingsFrame.pack(expand='true', fill='both', ipadx='5', ipady='5', side='bottom')
        self.OptionsFrame.configure(height='200', width='200')
        self.OptionsFrame.pack(expand='false', fill='both', ipadx='5', ipady='5', padx='5', pady='5', side='left')

        self.RightFrame = tkinter.ttk.Frame(self.MainPaneFrame)
        self.CameraPreviewFrame = tkinter.ttk.Frame(self.RightFrame)

        self.CameraPreviewCanvas = tkinter.Canvas(self.CameraPreviewFrame)
        self.CameraPreviewCanvas.pack(expand='true', fill='both', padx='5', pady='5', side='top')
        self.CameraPreviewFrame.configure(height='200', width='200')
        self.CameraPreviewFrame.pack(expand='true', fill='both', side='top')

        self.LaunchFrame = tkinter.ttk.Labelframe(self.RightFrame)

        self.ConstPoseButton = tkinter.ttk.Button(self.LaunchFrame)
        self.ConstPoseButton.configure(text='Constant Pose Mode')
        self.ConstPoseButton.pack(expand='true', side='left')
        self.ConstPoseButton.configure(command=self.ConstPoseCallback)

        self.ConstantCursorButton = tkinter.ttk.Button(self.LaunchFrame)
        self.ConstantCursorButton.configure(text='Constant Cursor Mode')
        self.ConstantCursorButton.pack(expand='true', side='right')
        self.ConstantCursorButton.configure(command=self.CursorModeCallback)

        self.LaunchFrame.configure(height='50', text='Launch', width='200')
        self.LaunchFrame.pack(expand='true', fill='both', ipadx='5', ipady='5', padx='5', pady='5', side='bottom')

        self.RightFrame.configure(height='200', width='200')
        self.RightFrame.pack(expand='true', fill='both', side='right')

        self.MainPaneFrame.configure(height='700', width='600')
        self.MainPaneFrame.pack(expand='true', fill='both', side='top')

        self.Window.add(self.MainPaneFrame, weight='1')
        self.StatusBarFrame = tkinter.ttk.Frame(self.Window)
        self.StatusBarOutputText = tkinter.Text(self.StatusBarFrame)
        self.StatusBarOutputText.configure(autoseparators='false', height='3', insertunfocussed='none', setgrid='false')
        self.StatusBarOutputText.configure(state='disabled', tabstyle='tabular', undo='false', width='50')
        _text_ = '''1Status messages ...
2Status messages ...
3Status messages ...
4Status messages ...
5Status messages ...
6Status messages ...
7Status messages ...
8end'''
        self.StatusBarOutputText.configure(state='normal')
        self.StatusBarOutputText.insert('0.0', _text_)
        self.StatusBarOutputText.configure(state='disabled')
        self.StatusBarOutputText.pack(expand='true', fill='both', side='left')

        self.StatusBarFrame.configure(height='30', width='200')
        self.StatusBarFrame.pack(side='right')
        self.Window.add(self.StatusBarFrame, weight='0')
        self.Window.configure(height='600', width='800')
        self.Window.pack(side='top')

        # Main widget
        self.main = self.Window

    def ChangePathCallback(self, event=None):
        self.path.set(tkinter.filedialog.askdirectory(initialdir=self.path.get(), mustexist=False,
                                                      title="Choose directory for storing data"))

    def ConstPoseCallback(self, event=None):
        recording_mode(mode='pose')

    def CursorModeCallback(self, event=None):
        pass

    def ShowPreviewCallback(self, event=None):
        pass

    def HidePreviewCallback(self, event=None):
        pass


def recording_mode(mode: str = None, path: str = None):
    global root
    gui = AcquisitionGui.gui
    gui.recordingMode = True
    gui.Window.pack_forget()
    # tkinter.PanedWindow.pack_forget()
    # time.sleep(2)
    # gui.Window.pack()
    root.attributes('-fullscreen', True)
    root.attributes('-alpha', 0.8)
    root.overrideredirect(True)
    # print("hiding window")


def gui_mode():
    global root
    gui = AcquisitionGui.gui
    gui.recordingMode = False
    gui.Window.pack()
    # tkinter.PanedWindow.pack_forget()
    # time.sleep(2)
    # gui.Window.pack()
    root.attributes('-fullscreen', False)
    root.attributes('-alpha', 1)
    root.overrideredirect(False)


def key_handler(event):
    # Replace the window's title with event.type: input key
    root.title("{}: {}".format(str(event.type), event.keysym))
    ctrl_pressed = (event.state & 0x4) != 0
    if event.keysym == 'Escape':
        if AcquisitionGui.gui.recordingMode:
            gui_mode()
        else:
            video_capture.stop_capture()
            cv.destroyAllWindows()
            root.destroy()
    elif event.keysym == 'p':
        if ctrl_pressed:
            print("ctrl p")
    elif event.keysym == 'm':
        if ctrl_pressed:
            print("ctrl m")
    elif event.keysym == 'b':
        if ctrl_pressed:
            print("ctrl b")


def render_preview():
    global root
    gui = AcquisitionGui.gui

    if gui.showPreview.get():
        root.update()
        height = gui.CameraPreviewCanvas.winfo_height()
        width = gui.CameraPreviewCanvas.winfo_width()
        root.img = video_capture.get_frame_tk(height,
                                              width)  # has to be declared here, or else garbage collector eats it

        y_offset = int((height - root.img.height()) / 2)
        x_offset = int((width - root.img.width()) / 2)

        gui.CameraPreviewCanvas.create_image(x_offset, y_offset, anchor=tkinter.NW, image=root.img)
    else:
        gui.CameraPreviewCanvas.delete('all')

    # print("rendering now")
    root.after(video_capture.wait_period, render_preview)


def main() -> None:
    global root
    root = tkinter.Tk()
    AcquisitionGui(root)

    # root.title("Oculotracker Training Data Acquisition Tool")
    # root.geometry(200,300)
    # root.attributes('-fullscreen', True)
    # root.attributes('-alpha', 0.8)
    # root.overrideredirect(1)
    root.bind('<KeyPress>', key_handler)
    video_capture.start_capture()
    root.after(500, render_preview)

    root.mainloop()
    video_capture.stop_capture()


if __name__ == "__main__":
    main()
