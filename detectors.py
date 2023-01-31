from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.point_rend import add_pointrend_config

from PIL import Image
import numpy as np
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PanelDetector():
    def __init__(self, panel_config, panel_checkpoint, panel_threshold=0.88, model_device='cpu'):
        self.panel_model = None
        self.panel_classes = None
        self.panel_class_metadata = None
        self.panel_config = panel_config
        self.panel_checkpoint = panel_checkpoint
        self.panel_threshold = panel_threshold
        self.model_device = model_device
        self.initialise_panel()

    def initialise_panel(self):

        cfg = get_cfg()
        add_pointrend_config(cfg)
        cfg.merge_from_file(self.panel_config)
        cfg.MODEL.WEIGHTS = self.panel_checkpoint
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.panel_threshold
        cfg.MODEL.DEVICE = self.model_device
        self.panel_model = DefaultPredictor(cfg)
        logger.info('Panel Model Loaded')
    
    def predict(self, img):
        '''
        Function that takes in a Numpy Array and country code and returns an Masked Image or None
        '''
        t1 = time.time()
        predictions = self.panel_model(img)
        t2 = time.time()
        logger.info('Panel Detection Model Inference took: {}'.format(str(t2 - t1)))

        instances = predictions["instances"].to("cpu")

        # Get detection boxes
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        pred_classes = instances.pred_classes.cpu().numpy()
        # If no boxes are detected, return original image and None for mask
        if len(boxes) == 0:
            return Image.fromarray(img), None
        else:
            # Crop Image
            return boxes, pred_classes

panel_detector = PanelDetector(
    panel_config='/home/ubuntu/Users/maixueqiao/carro-ds-cv-inspection/damage_localisation/configs/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml',
    panel_checkpoint='/home/ubuntu/Users/maixueqiao/carro-ds-cv-inspection/damage_localisation/model/model.pth',
    )