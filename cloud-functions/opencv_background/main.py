import cv2
import numpy as np


def image_circle_detection(input_path, output_path):
    img = cv2.imread(input_path)  # Image source
    if img is None:  # Check if image exists
        print('Error opening image!')
        print('Usage: Hough_Circle.py [image_name -- default ' + default_file + '] \n')
        print("Fail")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to gray
    gray = cv2.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 4,
                               param1=200, param2=60,
                               minRadius=0, maxRadius=0)  # Hough Circle function
    mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)  # Mask creation
    circle_details = None
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)  # Circle on mask around gauge
            circle_details = i
        fg = cv2.bitwise_or(img, img, mask=mask)  # Compares img to mask
        mask = cv2.bitwise_not(mask)
        background = np.full(img.shape, 255, dtype=np.uint8)
        bk = cv2.bitwise_or(background, background, mask=mask)
        bk[mask == 255] = (255, 255, 255)  # Set color of mask background
        final = cv2.bitwise_or(fg, bk)

        cropped = final[circle_details[1] - circle_details[2]:circle_details[1] + circle_details[2],
                  circle_details[0] - circle_details[2]:circle_details[0] + circle_details[2]]
        finalR = cv2.resize(cropped, (1920, 1080))  # Set img size
        cv2.imshow("circle mask", finalR)  # Show image (Future location to be a Google Storage bucket
        cv2.imwrite(output_path, finalR)
    else:
        print("No Circles Found")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image = ''  # Place path to image here
    path = ''  # Place path to save new image to here
    image_circle_detection(image, path)


if __name__ == '__main__':
    main()
