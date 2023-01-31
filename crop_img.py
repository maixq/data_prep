import numpy as np
from PIL import Image
import cv2
import math 
from models import panel_detector

class Cropper:
    
    def __init__(self, img, dst, detector, class_d):
        self.img_name = img.split('/')[-1] 
        self.image = np.array(Image.open(img).convert('RGB'))
        self.dst = dst
        self.detector = detector
        self.class_d = class_d

    def get_panel(self):
        panel_boxes, pred_classes = self.detector.predict(self.image)

        for box_idx in range(len(panel_boxes.tolist())):
            crop_panel = [k for k,v in self.class_d.items() if v == pred_classes[box_idx]][0]
            if (crop_panel.split('_')[-1] == 'headlight') or (crop_panel.split('_')[-1] == 'plate') :
                pass
            else:
                crop_pts = panel_boxes[box_idx]
                crop_img = Image.fromarray(self.image).crop((crop_pts[0], crop_pts[1], crop_pts[2], crop_pts[3]))
                print('IMG SIZE: ', crop_img.size)
                crop_img.save(self.dst+'{}_{}'.format(crop_panel, self.img_name))
                # crop_img = cv2.cvtColor(np.array(crop_img), cv2.COLOR_BGR2RGB)
                # print(crop_panel)
                # Image.fromarray(crop_img).save(self.dst+'{}_{}'.format(crop_panel, self.img_name))
        return

if __name__ == "__main__":
    panel_classes = {
                'car_license_plate': 0,
                'front_bonnet': 1,
                'front_bumper': 2,
                'front_left_bonnet': 3,
                'front_left_bumper': 4,
                'front_left_door': 5,
                'front_left_fender': 6,
                'front_left_headlight': 7,
                'front_left_side_mirror': 8,
                'front_right_bonnet': 9,
                'front_right_bumper': 10,
                'front_right_door': 11,
                'front_right_fender': 12,
                'front_right_headlight': 13,
                'front_right_side_mirror': 14,
                'rear_bonnet': 15,
                'rear_bumper': 16,
                'rear_left_bonnet': 17,
                'rear_left_bumper': 18,
                'rear_left_door': 19,
                'rear_left_fender': 20,
                'rear_left_headlight': 21,
                'rear_left_side_panel': 22,
                'rear_right_bonnet': 23,
                'rear_right_bumper': 24,
                'rear_right_door': 25,
                'rear_right_fender': 26,
                'rear_right_headlight': 27,
                'rear_right_side_panel': 28
            }

    img_path = '/home/ubuntu/Users/maixueqiao/damage_detection/jeff/detectron2/projects/PointRend/data/images/val/carro_0tBHrYmBkkAn8HzI.jpg'
    dst = '/home/ubuntu/Users/maixueqiao/data_prep/cropped_img/'

    cropper = Cropper(img_path, dst, panel_detector, panel_classes)
    cropper.get_panel()