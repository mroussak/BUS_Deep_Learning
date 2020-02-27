from mrcnn import utils
import numpy as np
import cv2
import glob
import os

class BUS_Dataset(utils.Dataset):
    
    def __init__(self):
        super().__init__()
    
    def load_data(self, class_names, path_to_img_dir):
        for idx, class_name in enumerate(class_names):
            self.add_class('local', idx+1, class_name)
        for idx, img in enumerate(os.listdir(path_to_img_dir)):
            self.add_image('local',image_id=idx,path=os.path.join(path_to_img_dir,img))
            
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
#         print(self.image_info[image_id]['path'])
        image = cv2.imread(self.image_info[image_id]['path'],cv2.IMREAD_GRAYSCALE)      
        image = image[:,:, np.newaxis] #Add 1 dimension for grayscale images
        return image
            
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask_paths = glob.glob(info['path'].replace('images', 'masks').replace('.png', '*.png'))
        masks = []
        class_ids = []
        for mask_path in mask_paths:
#             print(mask_path)
            mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)  
            masks.append(mask)
            if 'normal' in mask_path:
                class_ids.append(0)
            if 'benign' in mask_path:
                class_ids.append(1)
            if 'malignant' in mask_path:
                class_ids.append(2)
        masks = np.moveaxis(masks,0,-1)
        class_ids = np.array(class_ids)
        return masks, class_ids
    
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info['source'] == 'local':
            return info['source']
        else:
            super(self.__class__).image_reference(self, image_id)