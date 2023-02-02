import numpy as np
from PIL import Image
from detectors import panel_detector
from dictionary import panel_classes

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import os
from pycocotools.coco import COCO
import json
import math

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--dst', 
    help="dir destination ")
parser.add_argument( '--img_source', 
    help="image dir destination ")
parser.add_argument( '-i', '--json_source', 
    help="coco json source")

args = parser.parse_args()

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

class augmentAnnCropper:
    '''
    Augment annotation json for cropped panel images
    '''

    def __init__(self, json_path): 
        self.coco_annotation = COCO(annotation_file=json_path)
        self.seed = 42
    
    def augment_image(self, image, keypoints, seed, cropped, ht, wd):
        '''
        augment image and keypoints
        INPUT:
            image: original image
            keypoints: keypoints of panel
        OUTPUT:
            image_aug: image after augmentation
            kps_aug: keypoints after augmentation
        '''
        ia.seed(seed)
        cropx, cropy, w, h = cropped[0], cropped[1], cropped[2] , cropped[3] 
        top, right, bottom, left = cropy, wd-w, ht-h, cropx
        seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.2)), # change brightness, doesn't affect keypoints
        iaa.Crop(px=((top), (right),(bottom),(left)), keep_size=False)
        ])
        
        # Augment keypoints and images.
        image_aug, kps_aug = seq(image=image,  keypoints=keypoints)
        kps_aug = kps_aug.clip_out_of_image_()

        return image_aug, kps_aug
    
    def convert_keypoints2coordinates(self, aug_keypoints):
        '''
        convert keypoints to xy coordinates
        INPUT:
            keypoints: new keypoints after augmentation
        OUTPUT:
            coordinates: xy coordinates [x,y]
        '''
        coordinates = []
        for i in range(len(aug_keypoints.keypoints)):
            newx = aug_keypoints.keypoints[i].x
            newy = aug_keypoints.keypoints[i].y
            coordinates.append([newx,newy])
        return coordinates


    def convert_coordinates2keypoints(self, coordinates):
        '''
        convert xy coordinates to keypoints which required by Keypoints on image
        INPUT:
            coordinates: xy coordinates of each panel 
        OUTPUT:
            keypoints: array of keypoints 
        '''
        keypoints = []
        for coor in coordinates:
            x = coor[0]
            y = coor[1]
            keypoints.append(Keypoint(x=x, y=y))
        return keypoints

    def get_annotation_lists(self, im_id, image, cropped, ht, wd, ct, filter_class):
        ann_ids = self.coco_annotation.getAnnIds(imgIds=[im_id], iscrowd=None)
        anns = self.coco_annotation.loadAnns(ann_ids)
        ann_per_img = []
        for instance in range(len(anns)):
            cropped_annotation = {}
            segmentation_list = []
            for polygon in range(len(anns[instance]['segmentation'])):
                per_damage = anns[instance]['segmentation'][polygon]

                bbox = anns[instance]['bbox']
                iscrowd = anns[instance]['iscrowd']
                area = anns[instance]['area']
                img_id = ct
                cat_id = anns[instance]['category_id']
                index = anns[instance]['id'] 
                coordinate_list = zip(per_damage[::2], per_damage[1::2])
                bb_x, bb_y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                
                keypoints = self.convert_coordinates2keypoints(coordinate_list)
                bbox_points = self.convert_coordinates2keypoints([[bb_x, bb_y], [bb_x+w, bb_y+h]])
                kpsoi = KeypointsOnImage(keypoints, shape=image.shape)
                bb_pts = KeypointsOnImage(bbox_points, shape=image.shape)
                image_aug, keypoints_aug= self.augment_image(image, kpsoi, self.seed, cropped, ht, wd)

                image_aug, bbpoints_aug= self.augment_image(image, bb_pts, self.seed, cropped, ht, wd)
                coordinates_aug = self.convert_keypoints2coordinates(keypoints_aug)

                bbox_aug = self.convert_keypoints2coordinates(bbpoints_aug)
                crop_seg = [x for pair in coordinates_aug for x in pair]

                segmentation_list.append(crop_seg)
                new_bb = [x for pair in bbox_aug for x in pair]
                if bbox_aug and crop_seg and (len(new_bb) == 4) and (cat_id == filter_class): 
                    new_bb_x, new_bb_y, new_w, new_h = new_bb[0], new_bb[1], new_bb[2]-new_bb[0], new_bb[3]-new_bb[1]
                    new_bbox = [new_bb_x, new_bb_y, new_w, new_h]

                # write into dictionary
                    cropped_annotation['segmentation'] = segmentation_list
                    cropped_annotation['area'] = area
                    cropped_annotation['iscrowd'] = iscrowd
                    cropped_annotation['image_id'] = img_id
                    cropped_annotation['bbox'] = new_bbox
                    cropped_annotation['category_id'] = cat_id
                    cropped_annotation['id'] = index
                else:
                    pass

            if cropped_annotation:
                ann_per_img.append(cropped_annotation)
        return ann_per_img, image_aug

class Helper:

    def read_coco(self, json_path):
        with open (json_path, 'r') as f:
            coco = json.load(f)
            info = coco['info']
            licenses = coco['licenses']
            images = coco['images']
            annotations = coco['annotations']
            categories = coco['categories']
        return info, licenses, images, annotations, categories

    def save_coco(self, file, info, licenses, images, annotations, categories):
        with open(file, 'wt', encoding='UTF-8') as coco:
            json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
                'annotations': annotations, 
                'categories': categories},
                coco, indent=2, sort_keys=False)
        return

if __name__ == "__main__":

    # img_path = '/home/ubuntu/Users/maixueqiao/damage_detection/jeff/detectron2/projects/PointRsend/data/images/val/carro_0tBHrYmBkkAn8HzI.jpg'
    # cropper = Cropper(img_path, dst, panel_detector, panel_classes)
    # cropper.get_panel()

    json_path = args.json_source
    img_dir = args.img_source
    dst = args.dst

    helper = Helper()
    aug_cropper = augmentAnnCropper(json_path=json_path) 
    sample = os.listdir(img_dir)
    info, licenses, images, annotations, categories = helper.read_coco(json_path)

    ann_list = []
    im_list = []
    count = 0

    for i, im in enumerate(images[:10]):
        panel_pt_dic = {}
        filename = im['file_name']

        for x, sam in enumerate(sample):
            if filename == sample[x]:
                im_id = im['id']
                img_path = img_dir+'/'+sample[x]
                image = np.array(Image.open(img_path).convert('RGB'))
                ht, wd = image.shape[0], image.shape[1]
                panel_boxes, pred_classes = panel_detector.predict(image)

                try:
                    for box_idx in range(len(panel_boxes.tolist())):
                        img_d = {}
                        crop_panel = [k for k,v in panel_classes.items() if v == pred_classes[box_idx]][0]
                        if (crop_panel.split('_')[-1] == 'headlight') or (crop_panel.split('_')[-1] == 'plate') :
                            pass
                        else:
                            crop_pts = panel_boxes[box_idx]
                            crop_img = Image.fromarray(image).crop((crop_pts[0], crop_pts[1], crop_pts[2], crop_pts[3]))
                            crop_cor = [math.ceil(x) for x in crop_pts]
                            crop_w, crop_h =  crop_img.size
                            print('IMG SIZE: ', crop_img.size)
                            annotation_per_img, img_aug = aug_cropper.get_annotation_lists(im_id, image, crop_cor, ht, wd, count, filter_class=2)
                            if annotation_per_img:
                                img_d['file_name'] = crop_panel + '_' + im['file_name']
                                img_d['height'] = crop_h
                                img_d['width'] = crop_w
                                img_d['id'] = count
                                im_list.append(img_d)
                                ann_list.extend(annotation_per_img)
                                count+=1

                                if os.path.exists(dst+filename):
                                    print('Already exist!')
                                else:
                                    crop_img.save(args.dst+'/images/'+'{}_{}'.format(crop_panel, filename))
                except:
                    pass

    output_dir = args.dst + '/ann/' + 'test.json'
    helper.save_coco(output_dir, info, licenses, im_list, ann_list, categories)



