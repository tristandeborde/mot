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
    assert np.sum(im_back) == 16*3.


def test_interpol_boxes():
    b1 = [0, 0, 0, 0]
    b2 = [50, 50, 100, 100]
    num = 10
    assert video_visu.interpol_boxes(b1, b2, 0, num) == [0.0, 0.0, 0.0, 0.0]
    assert video_visu.interpol_boxes(b1, b2, 5, num) == [25., 25., 50., 50.]
    assert video_visu.interpol_boxes(b1, b2, num, num) == [50., 50., 100., 100.]


def test_process_tracking_result():
    tracking_result = {"detected_trash": [{'frame_to_box': {'212': [704.64, 478.08, 768., 514.56],
                                                            '213': [721.92, 476., 789., 510.]},
                                            'id': 195,
                                            'label': 'fragments',}],
                        "fps": 4}
    visu = video_visu.VideoVisu(1920, 1080, 30, tracking_result)
    visu.process_tracking_result()
    idx = (212/4)*30+2
    assert idx in visu.frames_to_boxes_dict
    assert visu.frames_to_boxes_dict[idx][0]["label"] == "fragments"
    assert "coords" in visu.frames_to_boxes_dict[idx][0]


def test_scalebox():
    tracking_result = {"detected_trash": [], "fps": 4}
    visu = video_visu.VideoVisu(1920, 1080, 30, tracking_result)
    b = visu.scalebox([50, 50, 100, 100])
    assert b == [int(50 * 1920 / 1024), int(50 * 1080 / 768),
                 int(100 * 1920 / 1024), int(100 * 1080 / 768)]


def test_interpolate_trash_frames():
    tracking_result = {"detected_trash": [{'frame_to_box': {'212': [704.64, 478.08, 768., 514.56],
                                                            '213': [721.92, 476., 789., 510.]},
                                            'id': 195,
                                            'label': 'fragments',}],
                        "fps": 4}
    visu = video_visu.VideoVisu(1920, 1080, 30, tracking_result)
    new_frame_to_box = visu.interpolate_trash_frames(tracking_result["detected_trash"][0], 10)
    assert len(new_frame_to_box) == 10
    assert new_frame_to_box[0] == [int(212*10), [704.64, 478.08, 768., 514.56]]
    assert new_frame_to_box[-1][0] == int(213*10)-1


def test_draw_hud():
    im = np.ones((1080, 1920,3))
    hud_info = {"bottles":2, "fragments":1, "others":4, "update":True}
    tracking_result = {"detected_trash": [], "fps": 4}
    visu = video_visu.VideoVisu(1920, 1080, 30, tracking_result)
    visu.draw_hud(im, hud_info)


def test_draw_boxes():
    im = np.ones((1080, 1920,3))
    boxes = [{"coords":[704.64, 478.08, 768., 514.56], "label":"bottles"}]
    tracking_result = {"detected_trash": [], "fps": 4}
    visu = video_visu.VideoVisu(1920, 1080, 30, tracking_result)
    visu.draw_boxes(im, boxes)


def test_draw_all():
    im = np.ones((1080, 1920,3))
    tracking_result = {"detected_trash": [{'frame_to_box': {'212': [704.64, 478.08, 768., 514.56],
                                                            '213': [721.92, 476., 789., 510.]},
                                            'id': 195,
                                            'label': 'fragments',}],
                        "fps": 4}
    visu = video_visu.VideoVisu(1920, 1080, 30, tracking_result)
    visu.process_tracking_result()
    visu.draw_all(im, (212/4)*30+2)
