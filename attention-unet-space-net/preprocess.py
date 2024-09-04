from utils.data_loading import load_image
import cv2
import os
from PIL import Image
import numpy as np

def preprocess(pil_img, is_mask) :
    w, h = pil_img.size
    img = np.asarray(pil_img)

    if is_mask :
        mask_unique_vals = np.unique(img)
        non_zero_unique_vals = mask_unique_vals[mask_unique_vals != 0]

        # Set mask to 1 at positions of non-zero unique values
        mask = np.zeros((h, w), dtype=np.int64)
        for value in non_zero_unique_vals :
            mask[img == value] = 1

        return mask
    
    else : 
        image = img.transpose((2,0,1))
        
        return image

def modify_mask(image, mask) :
    # np.set_printoptions(threshold=np.inf)
    # Find the positions where all channels have the value 255
    white_pixel_mask = (image[0] == 255) & (image[1] == 255) & (image[2] == 255)

    # Get the coordinates of white pixels
    white_pixel_coords = np.argwhere(white_pixel_mask)
    
    print(len(white_pixel_coords))
    # print(mask)
    print(np.count_nonzero(mask==1))
    print('------------------------------------')
    # Filter 0s into the mask location
    for white_pixel_coord in white_pixel_coords : 
        mask[white_pixel_coord[0], white_pixel_coord[1]] = 0
    
    # print(mask)
    print(np.count_nonzero(mask==1))
    return 0 

if __name__ == '__main__' : 
    main_dir = '/home/kiss2024/data/Massachusetts_Roads_Dataset/tiff/'

    img_dir = os.path.join(main_dir, 'train')
    mask_dir = os.path.join(main_dir, 'train_labels')

    img_list = os.listdir(img_dir)

    for img_name in img_list :
        img_n = os.path.join(img_dir, img_name)
        mask_n = os.path.join(mask_dir,     img_name[:-1])

        print(img_n)
        print(mask_n)
        img = load_image(img_n)
        mask = load_image(mask_n)

        img = preprocess(img, is_mask=False)
        mask = preprocess(mask, is_mask=True)

        modify_mask(img, mask)
        assert False