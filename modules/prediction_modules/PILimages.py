from PIL import Image
from sklearn.datasets import load_files
import numpy as np
import os
import cv2


def load_dataset(path):
    data = load_files(path, load_content=False)
    guage_files = np.array(data['filenames'])

    gauge_categorie_names = np.array(data['target_names'])
    gauge_categories = np.array([gauge_categorie_names[i] for i in np.array(data['target'])])
    return guage_files, gauge_categories

def PIL_image(img_path, dest_path, gauge_cat, img_numb):
    # img_path='./data/testData/psi_5/20180723_170402(1).jpg'
    image = Image.open(img_path)
    image.thumbnail((400, 400))
    ##########
    # image.show()
    #########
    box = (22, 10, 265, 265)
    cropped_image = image.crop(box)
    #########
    # cropped_image.show()
    #########
    image_rot_90 = cropped_image.rotate(-90)
    ##
    dest_folder = dest_path + gauge_cat
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    dest_file = dest_folder + '/' + str(img_numb) + '.jpg'
    image_rot_90.save(dest_file)
    return

def PIL_all_images(imgs_path, dest_path, gauge_categories):
    for i in range(len(imgs_path)):
        PIL_image(img_path=imgs_path[i], dest_path=dest_path,
                  gauge_cat=gauge_categories[i], img_numb=i)
    return

guage_files, gauge_categories = load_dataset('./data/testData')
PIL_all_images(imgs_path=guage_files,
               dest_path='./data/testData_PILed/',
               gauge_categories=gauge_categories)


