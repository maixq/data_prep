import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import requests
from pycocotools.coco import COCO
import os 

class checkCOCO:
    
    def __init__(self, add_json, old_json):
        self.add_json = add_json
        self.old_json = old_json
        
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
            
    def compareCOCO(self):
        save_images, save_anns = list(), list()
        remove_dup = []
        info, licenses, new_images, new_annotations, new_categories = self.read_coco(self.add_json)
        info, licenses, old_images, old_annotations, old_categories = self.read_coco(self.old_json)
        coco_annotation = COCO(annotation_file=self.add_json)
        categories =  old_categories
        print(categories)
        
        if new_categories == old_categories:
            print('categories are the same')
        else:
            print('make categories the same')
            categories = old_categories
            
        new_im = [x['file_name'] for x in new_images]
        
        for old in old_images:
            if old['file_name'] in new_im:
                print(old['file_name'])
                remove_dup.append(old['file_name'])
            else:
                pass
        
        for new in new_images:
            if new['file_name'] not in remove_dup:
                save_images.append(new)
                ann_ids = coco_annotation.getAnnIds(imgIds=[new['id']], iscrowd=None)
                anns = coco_annotation.loadAnns(ann_ids)
                for ann in anns:
                    ann['category_id'] = 1
                save_anns.extend(anns)
            else:
                try:
                    os.remove('/home/ubuntu/Users/maixueqiao/data_prep/test/images/'+new['file_name'])
                    print('remove {}'.format(new['file_name']))
                except:
                    pass
 
        
        filter_new_json = '/home/ubuntu/Users/maixueqiao/data_prep/test/filter_new.json'
        self.save_coco(filter_new_json, info, licenses, save_images, save_anns, categories)
        
        
if __name__ == "__main__":
    
    new = '/home/ubuntu/Users/maixueqiao/data_prep/test/ann/test.json'
    old = '/home/ubuntu/Users/maixueqiao/data_prep/test/train-w-neg.json'
    checkcoco = checkCOCO(new, old)
    checkcoco.compareCOCO()
        
    