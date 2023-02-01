""" Manual initialisation of body location and orientation

NOTE: This code is horrible, but does what it needs to
"""

import os
import PIL
import numpy as np
from pynput import keyboard
import time

from motiontrack.utils import quaternion_multiply

from motiontrack.utils import euler_to_quaternion

def initialise(body,
               views,
               plots,
               image_files,
               X0 =None):

    if X0 is None:
        X0 = np.zeros(7)
    x0 = X0[0:3]
    Q0 = X0[3:7]

    global x
    global Q
    global loop
    global adjust_angle
    x = x0
    Q = Q0
    adjust_angle = False

    def update(xyz, q):

        body.update(xyz, q)
        body.plot()
        blobs = [view.get_blobs() for view in views]
        frames = [view.get_mesh() for view in views]

        for i, plot in enumerate(plots):
            plot.update_projection(blobs[i])
            plot.update_mesh(*frames[i])

        im_1 = PIL.Image.open(image_files[0])
        im_2 = PIL.Image.open(image_files[1])

        plots[0].update_image(np.array(im_1))
        plots[1].update_image(np.flip(np.array(im_2),axis=1))
        #  plots[1].update_image(np.array(im_2))

    loop = True
    coarse = 0.01
    fine = 0.001
    def on_press(key):
        global adjust_angle
        global loop
        if key == keyboard.KeyCode.from_char('q'):
            loop = False
        elif key == keyboard.Key.up:
            x[1] += coarse
        elif key == keyboard.Key.down:
            x[1] -= coarse
        elif key == keyboard.Key.left:
            x[0] -= coarse
        elif key == keyboard.Key.right:
            x[0] += coarse
        elif key == keyboard.Key.page_up:
            x[2] += coarse
        elif key == keyboard.Key.page_down:
            x[2] -= coarse
        elif key == keyboard.KeyCode.from_char('i'):
            x[1] += fine
        elif key == keyboard.KeyCode.from_char('k'):
            x[1] -= fine
        elif key == keyboard.KeyCode.from_char('j'):
            x[0] -= fine
        elif key == keyboard.KeyCode.from_char('l'):
            x[0] += fine
        elif key == keyboard.KeyCode.from_char('u'):
            x[2] += fine
        elif key == keyboard.KeyCode.from_char('o'):
            x[2] -= fine
        elif key == keyboard.KeyCode.from_char('a'):
            adjust_angle += 1
        print(end="\r")
        listener.stop()
        #  print("\b \b \b \b")

    update(x, Q)

    print("----- STATE INITIALISATION ------")
    print("Using LOCAL coordinate frame")
    print("    -Arrow keys + xy coarse body movement")
    print("    -page up/page down for coarse z movement")
    print("    -(i),(j),(k),(l) keys for fine xy movement")
    print("    -(u),(o) for fine z movement")
    print("    -(a) to use GUI to change pose, press any key when complete")
    print("Press (Q) when initialisation complete")
    print("--------------------------------")
    while loop:
        with keyboard.Listener(on_press=on_press) as listener:
                listener.join()
        if adjust_angle:
            input("Adjust camera angle")
            dQ = body.get_camera_angle()
            Q = quaternion_multiply(Q,dQ)
            update(x, Q)
            adjust_angle -= 1
            print(end="\r")
        update(x, Q)
    X0[0:3] = x
    X0[3:7] = Q
    input("Press to continue")
    return X0


        #  if not result:
            #  loop = False
        #  print(_x, _y, _z, _q)
        #  print("----------------------------")
        #  update(np.array([_x, _y, _z]), _q)

