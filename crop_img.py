import numpy as np
from PIL import Image
from detectors import panel_detector
from dictionary import panel_classes

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
            # filter out headlight and license plate 
            if (crop_panel.split('_')[-1] == 'headlight') or (crop_panel.split('_')[-1] == 'plate') :
                pass
            else:
                crop_pts = panel_boxes[box_idx]
                crop_img = Image.fromarray(self.image).crop((crop_pts[0], crop_pts[1], crop_pts[2], crop_pts[3]))
                print('IMG SIZE: ', crop_img.size)
                crop_img.save(self.dst+'{}_{}'.format(crop_panel, self.img_name))
        return

if __name__ == "__main__":

    img_path = '/home/ubuntu/Users/maixueqiao/damage_detection/jeff/detectron2/projects/PointRend/data/images/val/carro_0tBHrYmBkkAn8HzI.jpg'
    dst = '/home/ubuntu/Users/maixueqiao/data_prep/cropped_img/'

    cropper = Cropper(img_path, dst, panel_detector, panel_classes)
    cropper.get_panel()