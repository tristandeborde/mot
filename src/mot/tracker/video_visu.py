import cv2
import wget
import os

def get_icons():
    """Loads the icons (and download them if they're unavailable)
    """
    ICONS = {
        "http://files.heuritech.com/raw_files/surfrider/bottle.png" : ".mot/resources/bottle.png",
        "http://files.heuritech.com/raw_files/surfrider/fragment.png" : ".mot/resources/fragment.png",
        "http://files.heuritech.com/raw_files/surfrider/other.png" : ".mot/resources/other.png"
    }

    home = os.path.expanduser("~")
    if not os.path.isdir(os.path.join(home, ".mot/")):
        os.mkdir(os.path.join(home, ".mot/"))
    if not os.path.isdir(os.path.join(home, ".mot/resources")):
        os.mkdir(os.path.join(home, ".mot/resources"))

    for k,v in ICONS.items():
        path = os.path.join(home, v)
        if not os.path.isfile(path):
            wget.download(k, path)
            print("\ndownloaded to ", path)
    return [cv2.imread(filename,-1) for filename in [os.path.join(home, ".mot/resources/bottle.png"),
                                                     os.path.join(home, ".mot/resources/fragment.png"),
                                                     os.path.join(home, ".mot/resources/other.png")]]


def overlay_im_to_background(im_back, im_over, x_offset, y_offset):
    """Adds an overlay image on a background image, including alpha channel

    Arguments:

    - im_back: background image (numpy array extracted from video)
    - im_over: image to overlay (numpy array in BGRalpha, to open with cv2.imread("path", -1)
    - x_offset: position of the topleft corner of the overlaid image
    - y_offset: position of the topleft corner of the overlaid image

    Returns:

    - Nothing, modifies inplace the image

    """
    y1, y2 = y_offset, y_offset + im_over.shape[0]
    x1, x2 = x_offset, x_offset + im_over.shape[1]

    alpha_s = im_over[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        im_back[y1:y2, x1:x2, c] = (alpha_s * im_over[:, :, c] +
                                    alpha_l * im_back[y1:y2, x1:x2, c])

def interpol_boxes(b1, b2, i, num):
    """Interpolates between boxes

    Arguments:

    - b1: bounding box 1 (array of coordinates)
    - b2: bounding box 2 (array of coordinates)
    - i: integer index of interpolation between 0 and num
    - num: total number of interpolations needed

    Returns:

    - A bounding box interpolated between b1 and b2 at position i/num
    """
    return [b1x + (b2x - b1x) * i * 1.0 / num for b1x, b2x in zip(b1, b2)]

class VideoVisu():
    def __init__(self, video_w, video_h, video_fps, tracking_result):
        """Visualisation tool to overlay tracking information to videos
        This is mainly a cosmetic class to have a clean output visualisation
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.thickness = 2
        self.thicknessUpdate = 3
        self.color = (238, 221, 192) # A surfrider color
        self.icons = get_icons()
        self.classes_to_icons = {'bottles':self.icons[0], 'fragments':self.icons[1], 'others':self.icons[2]}
        self.video_w = video_w
        self.video_h = video_h
        self.video_fps = video_fps
        self.tracking_result = tracking_result
        self.detection_image_size = (1024, 768)
        self.frames_to_boxes_dict = None
        self.frames_to_update_hud = None

    def process_tracking_result(self):
        """Main function which processes the tracking result to:
        - add interpolation between frames of detected objects to match the
        fps of the original video
        - groups information per frame

        Arguments:

        - None, it simply uses the tracking result defined in the __init__

        Returns:

        - Nothing, fills the two dictionnaries `frames_to_boxes_dict` and `frames_to_update_hud`

        """
        fps_ratio = self.video_fps / self.tracking_result["fps"]
        frames_to_boxes_dict = {}
        self.frames_to_update_hud = {}
        last_hud_info = {"bottles": 0, "fragments":0, "others":0}

        for trash_result in self.tracking_result["detected_trash"]:
            # Place the interpolated frames into frames_to_boxes_dict
            interpolated_frames = self.interpolate_trash_frames(trash_result, fps_ratio)
            for [idx, box] in interpolated_frames:
                if idx in frames_to_boxes_dict:
                    frames_to_boxes_dict[idx].append({"coords": box, "label":trash_result["label"]})
                else:
                    frames_to_boxes_dict[idx] = [{"coords": box, "label":trash_result["label"]}]

            # update the hub
            idx = interpolated_frames[0][0]
            label = trash_result["label"]
            new_hud_info = last_hud_info.copy()
            new_hud_info[label] += 1
            self.frames_to_update_hud[idx] = new_hud_info

        self.frames_to_boxes_dict = frames_to_boxes_dict

    def interpolate_trash_frames(self, trash, fps_ratio):
        """Interpolates between frames of a detected trash with a given fps ratio

        Arguments:

        - trash: A dictionnary of a detected trash. Must contain the "frame_to_box" key
        - fps_ratio: a floating point value giving the fps ratio between the original video and the detection

        Returns:

        - A list of [frame ids, box]

        """
        new_frame_to_box = []
        # For each frame in the detected trash
        for k,v in sorted(trash["frame_to_box"].items(), key=lambda x:x[0]):
            new_idx = int(int(k)*fps_ratio)
            if new_frame_to_box:
                [old_idx, old_v] = new_frame_to_box[-1]
                print([old_idx, old_v])
                for i,idx in enumerate(range(old_idx+1, new_idx)):
                    new_frame_to_box.append([idx,interpol_boxes(old_v, v, i+1, new_idx-old_idx)])
            else:
                new_frame_to_box.append([new_idx,v])
        return new_frame_to_box

    def scalebox(self, b):
        """scales a boundnig box (output of the detection process)
        to match the original video scale

        Arguments:

        -b: box = array of 4 coordinates (x1, y1, x2, y2)
        """
        return [int(b[0]*self.video_w/self.detection_image_size[0]),
                int(b[1]*self.video_h/self.detection_image_size[1]),
                int(b[2]*self.video_w/self.detection_image_size[0]),
                int(b[3]*self.video_h/self.detection_image_size[1])]

    def draw_hud(self, im, hud_info):
        """Draws the hud (heads-on display) on top of the background image2

        Arguments:

        - im: background image (numpy array extracted from video)
        - hud_info: dictionnary with the counts of each class, and the key "update"
        if the current frame got an update

        Returns:

        - Nothing, modifies inplace the image
        """
        # Black background of hud
        cv2.rectangle(im,(30,30),(200,200),(0,0,0),-1)
        # Add icons
        overlay_im_to_background(im, self.classes_to_icons["bottles"], 65,40)
        overlay_im_to_background(im, self.classes_to_icons["fragments"], 60,95)
        overlay_im_to_background(im, self.classes_to_icons["others"], 60,150)
        # Add counts
        thickness = self.thicknessUpdate if "update" in hud_info else self.thickness
        cv2.putText(im, str(hud_info["bottles"]), (110,75), self.font, self.fontScale, self.color, thickness, cv2.LINE_AA)
        cv2.putText(im, str(hud_info["fragments"]), (110,125), self.font, self.fontScale, self.color, thickness, cv2.LINE_AA)
        cv2.putText(im, str(hud_info["others"]), (110,175), self.font, self.fontScale, self.color, thickness, cv2.LINE_AA)

    def draw_boxes(self, im, boxes):
        """Draws the bounding boxes and icons next to items

        Arguments:

        - im: background image (numpy array extracted from video)
        - boxes: a list of boxes [{"coords": [], "label":"bottles"}]

        Returns:

        - Nothing, modifies inplace the image
        """
        for bbox in boxes:
            l = [int(x) for x in bbox["coords"]]
            l = self.scalebox(l)
            icon = self.classes_to_icons[bbox["label"]]
            overlay_im_to_background(im, icon, l[0], l[1] - icon.shape[0] - 5)
            cv2.rectangle(im,(l[0],l[1]),(l[2],l[3]),self.color,2)

    def draw_all(self, im, idx):
        """Draws everything needed on a video frame at given idx.
        Assumes frames_to_boxes_dict and frames_to_update_hud are defined by
        having run `process_tracking_result`

        Arguments:

        - im: image (numpy array)
        - idx: frame index of video

        Returns:

        - Nothing, modifies the image inplace
        """
        if idx in self.frames_to_boxes_dict:
            self.draw_boxes(im, self.frames_to_boxes_dict[idx])

        hud_info = {"bottles": 0, "fragments":0, "others":0}
        if idx in self.frames_to_update_hud:
            hud_info = self.frames_to_update_hud[idx].copy()
            hud_info["update"] = True
        else:
            # find the last hud info
            for x in self.frames_to_update_hud.keys():
                if idx < x:
                    break
                hud_info = self.frames_to_update_hud[x]
        self.draw_hud(im, hud_info)
