import os
import shutil

import numpy as np

from mot.tracker import video_visu

def test_get_icons():
    icons = video_visu.get_icons()
    assert len(icons)==3
    assert len(icons[0].shape) == 3

def test_overlay_im_to_background():
    im_back = np.zeros((10,10,3))
    im_over = np.ones((4,4,4))
    im_over[:,:,3] = 255
    x_offset = 1
    y_offset = 2
    video_visu.overlay_im_to_background(im_back, im_over, x_offset, y_offset)
    assert np.sum(im_back) == 16

def test_interpol_boxes():
    b1 = [0, 0, 0, 0]
    b2 = [50, 50, 100, 100]
    num = 10
    assert video_visu.interpol_boxes(b1, b2, 0, num) == [0.0, 0.0, 0.0, 0.0]
    assert video_visu.interpol_boxes(b1, b2, 5, num) == [25., 25., 50., 50.]
    assert video_visu.interpol_boxes(b1, b2, num, num) == [50., 50., 100., 100.]

def test_video_visu(tmpdir):
    tracking_result = {"detected_trash": [], "fps": 4}
    visu = video_visu.VideoVisu(1920, 1080, 30, tracking_result)
