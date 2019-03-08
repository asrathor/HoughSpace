import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the images
img1 = cv2.imread('point.jpg',0)
img2 = cv2.imread('segment.jpg',0)

# This function draws something similar to histogram, however instead of bins I can know the number
# of each pixels in image and represent them as a curve in occurence-values graph.
def histogram(img_arr):

    temp = [0 for x in range(256)]

    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            for k in range(0,256):
                if img_arr[i][j] == k:
                    temp[k] = temp[k] + 1
                    break
    return temp

histt = histogram(img2)
plt.plot(histt)
plt.savefig('task2b_occurences_curve.jpg')

# The structuring element for point detection.
def kernel_struct():

    kernel = np.ones((3,3))
    kernel[0][0] = -1
    kernel[0][1] = -1
    kernel[0][2] = -1
    kernel[1][0] = -1
    kernel[1][1] = 8
    kernel[1][2] = -1
    kernel[2][0] = -1
    kernel[2][1] = -1
    kernel[2][2] = -1

    return kernel

# This function is used to pad the image by zeros.
def padding(matrixx):

    final_mat = np.zeros((matrixx.shape[0]+2,matrixx.shape[1]+2))

    for i in range(matrixx.shape[0]):
        for j in range(matrixx.shape[1]):
            final_mat[i+1][j+1] = matrixx[i][j]

    return final_mat

# This function is used to slice the image in 3,3 matrix.
def slice_mat(matrixx,index_x,index_y):

    mat = np.zeros((3,3))
    mat[0][0] = matrixx[index_x][index_y]
    mat[0][1] = matrixx[index_x][index_y+1]
    mat[0][2] = matrixx[index_x][index_y+2]
    mat[1][0] = matrixx[index_x+1][index_y]
    mat[1][1] = matrixx[index_x+1][index_y+1]
    mat[1][2] = matrixx[index_x+1][index_y+2]
    mat[2][0] = matrixx[index_x+2][index_y]
    mat[2][1] = matrixx[index_x+2][index_y+1]
    mat[2][2] = matrixx[index_x+2][index_y+2]

    return mat

# A correlation is performed by multiplying the values pixels by pixels and summing them up.
# This algorithm is referred from lecture slides.
def point_analysis(sliced_matrix):

    kernel = kernel_struct()
    summ = 0
    for i in range(sliced_matrix.shape[0]):
        for j in range(sliced_matrix.shape[1]):

            summ = summ + (sliced_matrix[i][j]*kernel[i][j])
    # A point of interest is detected specifically when threshold (representing sum) is equal to 339.
    # Since the threshold in this case is for sum, it was calculated manually.
    if summ == 339:
        return True
    else:
        return False

# This function is used to perform point detection and place the coordinates of point on the image.
def point_detection():

    padded_img = padding(img1)
    new_img = np.zeros((img1.shape[0],img1.shape[1]))
    counter = 0
    for i in range(padded_img.shape[0]-2):
        for j in range(padded_img.shape[1]-2):

            sliced_matrix = slice_mat(padded_img,i,j)

            # new_img[i][j] = point_analysis(sliced_matrix)

            if point_analysis(sliced_matrix) == True:

                new_img[i][j] = 255
                counter = counter+1

    y_axis = []
    x_axis = []

    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            if new_img[i][j] != 0:
                counter = counter + 1
                y_axis.append(i)
                x_axis.append(j)
    print('result:',new_img.shape)
    for i in range(len(y_axis)):
        if x_axis[i] < 20:
            cv2.putText(new_img, '(' + str(x_axis[i]) + ',' + str(y_axis[i]) + ')', (x_axis[i], y_axis[i]-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color=(255, 0, 0))
            cv2.circle(new_img,(x_axis[i],y_axis[i]),5,color=(255,0,0),thickness=1)
        if x_axis[i] > new_img.shape[1] - 20:
            cv2.putText(new_img, '(' + str(x_axis[i]) + ',' + str(y_axis[i]) + ')', (x_axis[i]-50, y_axis[i]-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color=(255, 0, 0))
            cv2.circle(new_img,(x_axis[i],y_axis[i]),5,color=(255,0,0),thickness=1)
        if (x_axis[i] > 200) and (x_axis[i] < new_img.shape[1] - 40):
            cv2.putText(new_img, '(' + str(x_axis[i]) + ',' + str(y_axis[i]) + ')', (x_axis[i]-10, y_axis[i]-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color=(255, 0, 0))
            cv2.circle(new_img,(x_axis[i],y_axis[i]),5,color=(255,0,0),thickness=1)
    print('counter:',counter)

    cv2.imwrite('point_detected.jpg',new_img)

# This function is used to perform segmentation.
def segmentation():
    # The value of T is determined by looking at the curve output from histogram function implemented above.
    T = 204
    count = 0
    obj = np.zeros(img2.shape)
    bkg = np.zeros(img2.shape)
    while count != 1:
        obj = np.zeros(img2.shape)
        bkg = np.zeros(img2.shape)

        for i in range(img2.shape[0]):
            for j in range(img2.shape[1]):

                if img2[i][j] > T:
                    obj[i][j] = img2[i][j]
                if img2[i][j] <= T:
                    bkg[i][j] = img2[i][j]

        u1 = obj.mean()
        print(u1)
        u2 = bkg.mean()
        print(u2)
        std1 = obj.std()
        std2 = bkg.std()
        print(std1)
        print(std2)
        #A = (std1 ** 2) - (std2 ** 2)
        #B = 2 * (u1 * (std2 ** 2) - u2 * (std1 ** 2))
        #P1 = 0.3
        #P2 = 0.7
        #C = ((std1 ** 2) * (u2 ** 2)) - ((std2 ** 2) * (u1 ** 2)) + (2 * (std1 ** 2) * (std2 ** 2) * np.log((std2 * P1)/(std1 * P2)))

        T = (u1+u2)/2
        count = count+1
        # print('error:',T)

    cv2.imwrite('segmentation.jpg',obj)
    return obj
    # plt.plot(obj)
    # plt.show()

# To draw one overall bounding box. However, since MS paint can be used to draw four boxes,
# this function was not used.
def bounding_box(img_arr):

    rows = img_arr.shape[0]
    cols = img_arr.shape[1]
    print(rows,cols)
    top_x = 0
    top_y = 0
    temp = [0 for x in range(cols)]
    for i in range(rows):
        for j in range(cols):
            temp[j] = img_arr[i][j]
        count = 0
        for k in range(cols):
            if temp[k] != 0:
                top_x = k
                count = count + 1
        print(count)
        if count >= 1:
            top_y = i
            break
    print(top_x,top_y)

    bot_x = 0
    bot_y = 0
    temp = [0 for x in range(cols)]
    for i in range(rows-1,-1,-1):
        for j in range(cols-1,-1,-1):
            temp[j] = img_arr[i][j]
        count = 0
        for k in range(cols):
            if temp[k] != 0:
                bot_x = cols - k
                count = count + 1
        # print(count)
        if count >= 1:
            bot_y = i
            break
    print(bot_x,bot_y)

    rig_x = 0
    rig_y = 0
    temp = [0 for x in range(rows)]
    for j in range(cols):
        for i in range(rows):
            temp[i] = img_arr[i][j]
        count = 0
        for k in range(rows):
            if temp[k] != 0:
                rig_y = k
                count = count + 1
        # print(count)
        if count >= 1:
            rig_x = j
            break
    print(rig_x, rig_y)

    lef_x = 0
    lef_y = 0
    temp = [0 for x in range(rows)]
    for j in range(cols-1,-1,-1):
        for i in range(rows):
            temp[i] = img_arr[i][j]
        count = 0
        for k in range(rows):
            if temp[k] != 0:
                lef_y = k
                count = count + 1
        # print(count)
        if count >= 1:
            lef_x = j
            break
    print(lef_x, lef_y)

    cv2.line(img_arr,(rig_x,top_y),(lef_x,top_y),(255,0,0),thickness=1)
    cv2.line(img_arr, (rig_x, top_y), (rig_x, bot_y), (255, 0, 0), thickness=1)
    cv2.line(img_arr, (lef_x, top_y), (lef_x, bot_y), (255, 0, 0), thickness=1)
    cv2.line(img_arr, (rig_x, bot_y), (lef_x, bot_y), (255, 0, 0), thickness=1)
    cv2.putText(img_arr,'('+str(rig_x)+','+str(top_y)+')',(rig_x-20,top_y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(lef_x) + ',' + str(top_y) + ')', (lef_x - 20, top_y - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(rig_x) + ',' + str(bot_y) + ')', (rig_x - 20, bot_y + 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(lef_x) + ',' + str(bot_y) + ')', (lef_x - 20, bot_y + 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, color=(255, 0, 0))

    cv2.imwrite('bounding_box.jpg',img_arr)

# Function to place the text on the image based on the coordinates of bounding box acquired from
# MS paint.
def put_text(img_arr):

    text_size = 0.5
    cv2.putText(img_arr, '(' + str(163) + ',' + str(126) + ')', (163 - 40, 126 - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(200) + ',' + str(126) + ')', (200 - 20, 126 - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(163) + ',' + str(163) + ')', (163 - 40, 163 + 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(200) + ',' + str(163) + ')', (200 - 20, 163 + 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))

    rig_x = 255
    top_y = 79
    lef_x = 301
    bot_y = 204
    cv2.putText(img_arr, '(' + str(rig_x) + ',' + str(top_y) + ')', (rig_x - 40, top_y - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(lef_x) + ',' + str(top_y) + ')', (lef_x - 20, top_y - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(rig_x) + ',' + str(bot_y) + ')', (rig_x - 40, bot_y + 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(lef_x) + ',' + str(bot_y) + ')', (lef_x - 20, bot_y + 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))

    rig_x = 336
    top_y = 26
    lef_x = 363
    bot_y = 285
    cv2.putText(img_arr, '(' + str(rig_x) + ',' + str(top_y) + ')', (rig_x - 40, top_y - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(lef_x) + ',' + str(top_y) + ')', (lef_x - 10, top_y - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(rig_x) + ',' + str(bot_y) + ')', (rig_x - 40, bot_y + 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(lef_x) + ',' + str(bot_y) + ')', (lef_x - 10, bot_y + 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))

    rig_x = 389
    top_y = 43
    lef_x = 421
    bot_y = 250
    cv2.putText(img_arr, '(' + str(rig_x) + ',' + str(top_y) + ')', (rig_x - 25, top_y - 5),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(lef_x) + ',' + str(top_y) + ')', (lef_x, top_y - 5),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(rig_x) + ',' + str(bot_y) + ')', (rig_x - 25, bot_y + 15),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))
    cv2.putText(img_arr, '(' + str(lef_x) + ',' + str(bot_y) + ')', (lef_x, bot_y + 15),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, color=(255, 0, 0))

    cv2.imwrite('Bounded_box_labeled.jpg',img_arr)


point_detection()
obj = segmentation()
bounded_img = cv2.imread('segmentation_Bound.jpg',0)
put_text(bounded_img)
# bounding_box(obj)




