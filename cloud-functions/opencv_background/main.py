import cv2
import numpy as np

def main():
    
    img = cv2.imread('/home/gabe/Documents/example_gauge.jpg') #test other images at diffferent angles/ false images
    #img = cv2.imread('/home/gabe/Downloads/Maurer.Gabe-1000-X2.jpg')
    if img is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        print("Fail")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 4,
                               param1=200, param2=100,
                               minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)

    if circles is not None:
        for i in circles[0, :]:
            cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)
    else:
        print("No Circles Found")

    fg = cv2.bitwise_or(img, img, mask=mask)
    mask = cv2.bitwise_not(mask)
    background = np.full(img.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)
    bk[mask == 255] = (255, 105, 147)
    final = cv2.bitwise_or(fg, bk)
    finalR = cv2.resize(final, (1920, 1080))
    cv2.imshow("circle mask", finalR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def dist_2_pts(x1, y1, x2, y2):
    #print np.sqrt((x2-x1)^2+(y2-y1)^2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def main1():
    cap = cv2.VideoCapture(0)
    t1 = True
    while(True):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        rows = frame.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 16,
                                  param1=200, param2=100,
                                  minRadius=0, maxRadius=0)

        mask = np.full((gray.shape[0], gray.shape[1]), 0, dtype=np.uint8)

        if circles is not None:
            print("found circle")
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(gray, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(gray, center, radius, (255, 0, 255), 3)






                """
                if i[0] is not 0 or i[1] is not 0 or i[2] is not 0:
                    print("in")
                    cv2.imwrite('/home/gabe/Documents/test_gauge.png', gray)
                    t1 = False
                    break
                else:
                    print("loc")
            if t1 == False:
                break
            """

            """
                # find lines
                minLineLength = 10
                maxLineGap = 0
                lines = cv2.HoughLinesP(image=gray, rho=3, theta=np.pi / 180, threshold=100, minLineLength=minLineLength,
                                        maxLineGap=0)  # rho is set to 3 to detect more lines, easier to get more then filter them out later4



                if lines is not None:
                    print("Found line")
                    final_line_list = []
                    # print "radius: %s" %r

                    diff1LowerBound = 0.15  # diff1LowerBound and diff1UpperBound determine how close the line should be from the center
                    diff1UpperBound = 0.25
                    diff2LowerBound = 0.5  # diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
                    diff2UpperBound = 1.0
                    for j in range(0, len(lines)):
                        for x1, y1, x2, y2 in lines[j]:
                            diff1 = dist_2_pts(i[0], i[1], x1, y1)  # x, y is center of circle
                            diff2 = dist_2_pts(i[0], i[1], x2, y2)  # x, y is center of circle
                            # set diff1 to be the smaller (closest to the center) of the two), makes the math easier
                            if (diff1 > diff2):
                                temp = diff1
                                diff1 = diff2
                                diff2 = temp
                            # check if line is within an acceptable range
                            if (((diff1 < diff1UpperBound * i[2]) and (diff1 > diff1LowerBound * i[2]) and (
                                    diff2 < diff2UpperBound * i[2])) and (diff2 > diff2LowerBound * i[2])):
                                line_length = dist_2_pts(x1, y1, x2, y2)
                                # add to final list
                                final_line_list.append([x1, y1, x2, y2])

                    # best line
                    x1 = final_line_list[0][0]
                    y1 = final_line_list[0][1]
                    x2 = final_line_list[0][2]
                    y2 = final_line_list[0][3]
                    cv2.line(gray, (x1, y1), (x2, y2), (255, 0, 0), 2)
                """

        else:
            print("No Circles Found")
            count = 0

        # Display the resulting frame
        cv2.imshow('frame',final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()