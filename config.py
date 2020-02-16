import os
import numpy as np
from mrcnn.config import Config
from pathlib import Path

len_training_set = len(os.listdir(Path('data/seg_dir/train_images').absolute()))
len_val_set = len(os.listdir(Path('data/seg_dir/val_images').absolute()))

class BUS_DS_Config(Config):
    """Configuration for training on the instance-segmentation dataset.
    Derives from the base Config class and overrides values specific
    to the instance-segmentation dataset.
    """
    # Give the configuration a recognizable name
    NAME = 'instance-seg-compact'
    
    BACKBONE = "resnet101"
    
    #Include these two configurations for grayscale images
    IMAGE_CHANNEL_COUNT = 1    
    MEAN_PIXEL = np.array([117.])

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 32

    # Number of classes (including background)
    NUM_CLASSES = 2  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 64*2
    IMAGE_MAX_DIM = 64*2

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    
    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = len_training_set/16

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = len_val_set/16
    
    DETECTION_MAX_INSTANCES = 1
    
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.8
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask