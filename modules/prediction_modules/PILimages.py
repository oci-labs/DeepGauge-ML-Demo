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


def Remove_Background_Open_CV():
    ##
    import numpy as np
    import cv2
    #
    ## for the needle
    img_path_0 = ('./logs/temporary_latest_streaming_image/2.png')
    imgFile_needle = cv2.imread(img_path_0)
    frame_needle = cv2.medianBlur(cv2.cvtColor(imgFile_needle, cv2.COLOR_BGR2GRAY), 5)
    ##

    img_path = ('./logs/temporary_latest_streaming_image/1.png')
    imgFile = cv2.imread(img_path)
    frame = cv2.medianBlur(cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY), 5)

    output = frame.copy()
    circles = cv2.HoughCircles(output, cv2.HOUGH_GRADIENT, minDist=200, dp=1.2,
                               param1=120, param2=50, minRadius=50, maxRadius=450)

    circles = np.round(circles[0, :]).astype("int")

    x = circles[0, 0]
    y = circles[0, 1]
    r = circles[0, 2]

    cv2.circle(output, (x, y), r + 240, (255, 255, 255), 500)
    # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # cv2.rectangle(output, (x - r, y - r), (x + r, y + r), (0, 128, 255), -1)
    ## to crop the original frame
    square_half_side = int(1.07 * float(r))
    # cropped_frame = output.copy()
    cropped_frame = output[x - square_half_side:x + square_half_side,
                    y - square_half_side:y + square_half_side]

    frame_needle_cropped = frame_needle[x - square_half_side:x + square_half_side,
                           y - square_half_side:y + square_half_side]

    img_path_save = ('./logs/temporary_latest_streaming_image/3.png')
    cv2.imwrite(img_path_save, cropped_frame)

    img_path_save_needle = ('./logs/temporary_latest_streaming_image/3_needle.png')
    cv2.imwrite(img_path_save_needle, frame_needle_cropped)

    return


guage_files, gauge_categories = load_dataset('./data/testData')
PIL_all_images(imgs_path=guage_files,
               dest_path='./data/testData_PILed/',
               gauge_categories=gauge_categories)
