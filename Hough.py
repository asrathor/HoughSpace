import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the images as both color and grayscale.
imgg = cv2.imread('hough.jpg')
img2_gray = cv2.imread('hough.jpg',0)

# The implemenation of sobel operator is taken from my project 1 submission.

# Kernel for x axis
Gx = [[0 for x in range(3)] for y in range(3)]
Gx[0][0] = -1
Gx[0][1] = 0
Gx[0][2] = 1
Gx[1][0] = -2
Gx[1][1] = 0
Gx[1][2] = 2
Gx[2][0] = -1
Gx[2][1] = 0
Gx[2][2] = 1

# Kernel for y axis
Gy = [[0 for x in range(3)] for y in range(3)]
Gy[0][0] = -1
Gy[0][1] = -2
Gy[0][2] = -1
Gy[1][0] = 0
Gy[1][1] = 0
Gy[1][2] = 0
Gy[2][0] = 1
Gy[2][1] = 2
Gy[2][2] = 1


# Pad the matrix by 0
def padding(matrixx):

    final_mat = np.zeros((matrixx.shape[0]+2,matrixx.shape[1]+2))

    for i in range(matrixx.shape[0]):
        for j in range(matrixx.shape[1]):
            final_mat[i+1][j+1] = matrixx[i][j]

    return final_mat


# The image is convoluted (flipped)
def convolution(matrixx):

    conv_mat = np.zeros((3,3))
    conv_mat[0][0] = matrixx[2][2]
    conv_mat[0][1] = matrixx[2][1]
    conv_mat[0][2] = matrixx[2][0]
    conv_mat[1][0] = matrixx[1][2]
    conv_mat[1][2] = matrixx[1][0]
    conv_mat[2][0] = matrixx[0][2]
    conv_mat[2][1] = matrixx[0][1]
    conv_mat[2][2] = matrixx[0][0]

    return conv_mat


# From a large matrix, a (3,3) matrix is sliced
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


# Implemenation of the main Sobel Operator. Taken from my project 1 submission.
def sobel_operator():

    rows = img2_gray.shape[0]+2
    columns = img2_gray.shape[1]+2
    comb_mat = np.zeros((rows,columns))
    x_img = np.zeros((rows,columns))
    y_img = np.zeros((rows,columns))
    img2_gray_padded = padding(img2_gray)

    for i in range(rows-2):
        for j in range(columns-2):

            slice_matrixx = slice_mat(img2_gray_padded,i,j)
            S1 = sum(sum(convolution(Gx) * slice_matrixx))
            S2 = sum(sum(convolution(Gy) * slice_matrixx))

            x_img[i+1][j+1] = S1
            y_img[i+1][j+1] = S2
            comb_mat[i + 1][j + 1] = ((S1 ** 2) + (S2 ** 2)) ** 0.5

    final_x_img = x_img/255
    final_y_img = y_img/255
    final_comb_img = comb_mat/255

    return comb_mat

# This segmentation function is the same as the one used in Task 2b. We need to segment the image after
# sobel filter to make the image as binary and further reduce the noise. This will make the process
# run faster.
def segmentation(img2, T):

    # T = 170
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
        # A = (std1 ** 2) - (std2 ** 2)
        # B = 2 * (u1 * (std2 ** 2) - u2 * (std1 ** 2))
        # P1 = 0.3
        # P2 = 0.7
        # C = ((std1 ** 2) * (u2 ** 2)) - ((std2 ** 2) * (u1 ** 2)) + (2 * (std1 ** 2) *
        # (std2 ** 2) * np.log((std2 * P1)/(std1 * P2)))

        T = (u1+u2)/2
        count = count+1
        # print('error:',T)

    return obj

# This function takes in the range of theta values to consider (for speeding up the task) and
# compute accumulator matrix from theta and ph (rho) values. A detailed explaination of this
# approach is explained in the project report.
def hough_transform(img_arr,angle1,angle2):

    theta = np.deg2rad(np.arange(angle1, angle2))
    diag_img = int(np.sqrt(img_arr.shape[0] ** 2 + img_arr.shape[1] ** 2)) + 1
    ph = np.array(np.arange(-diag_img, diag_img)).astype(int)
    print(len(ph))
    H = np.zeros((len(ph), len(theta)))

    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):

            if img_arr[i][j] != 0:
                for k in range(len(theta)):
                    d = int((j * np.cos(theta[k]) + i * np.sin(theta[k])) + diag_img)
                    H[d,k] = H[d,k] + 1

    return theta, ph, H

# Bonus Task (Not completed): This function computes the 3D matrix for a,b,radius based on theta and
# image coordinates.
def hough_transform_circle(img_arr):

    theta = np.deg2rad(np.arange(0,360))
    radius = np.array(np.arange(20,30,1)).astype(int)
    H = np.zeros((len(radius),img_arr.shape[1],img_arr.shape[0]))

    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):

            if img_arr[i][j] != 0:
                for k in range(len(radius)):
                    for l in range(0,len(theta),10):
                        a = int(j - radius[k] * np.cos(theta[l]))
                        b = int(i - radius[k] * np.sin(theta[l]))
                        H[k,a,b] = H[k,a,b] + 1

    return theta, radius, H

# This function computes the maximas in the accumulator matrix. The maximas extracted are the same as
# peaks required. If maximas are similar to each other, one of them is made as 0 to ensure that no lines
# very near to each other are drawn. A detailed explaination of this function is in the project report.
def peak_detection1(H, peaks):

    indices = np.argsort(H.flatten())
    temp = []
    count = 0
    for i in range(len(indices)-1,-1,-1):
        if count == peaks:
            break
        temp.append(indices[i])
        count = count + 1

    maxs = list(reversed(temp))

    # print(maxs)
    for i in range(len(maxs)):
        for j in range(len(maxs)):
            # print(abs(indices2[i] - indices2[j]))
            if abs(maxs[i] - maxs[j]) <= 1000 and maxs[i] != maxs[j]:
                maxs[i] = 0

    max_indexs = np.concatenate(np.unravel_index(maxs,H.shape)).reshape(2,-1).T

    return max_indexs

# This was the first try I was doing to extract maximas based on argmax function. However, even after
# debugging the extracted maximas very near to each other leading to overlapping lines. Thus argsort was
# used to get better line detection.
def peak_detection(H, peaks):

    temp = []
    count = 0
    while len(temp) != peaks:
        index2 = np.unravel_index(np.argmax(H, axis=None), H.shape)
        if count == 0:
            temp.append(index2)
            H[index2[0], index2[1]] = 0
        else:
            counter = 0
            print('temp:',temp)
            for i in range(len(temp)):
                y, x = temp[i]
                print('Reaching A',y,x)
                print('indexs:',index2[0],index2[1])
                abs_x = abs(x-index2[0])
                abs_y = abs(y - index2[1])
                if abs_x > 81:
                    diff_x = abs_x - 81
                else:
                    diff_x = 81 - abs_x
                if abs_y > 81:
                    diff_y = abs_y - 81
                else:
                    diff_y = 81 - abs_y
                print(diff_x,diff_y)
                if diff_x >= 70 and diff_y >= 70:
                    print('Reaching B')
                    counter = 1
                else:
                    counter = 0
            if counter == 1:
                temp.append(index2)
            H[index2[0],index2[1]] = 0
        # if len(temp) == 0:
        #    temp.append(indices2)
        #    H[indices2[0], indices2[1]] = 0
        # else:
        #    counter = 0
        #    for i in range(len(temp)):
        #        y, x = temp[i]
        #        print('Reaching A',y,x)
        #        if abs(H[y][x] - H[indices2[0],indices2[1]]) >= 330:
        #            print('Reaching B')
        #            counter = 1
        #        else:
        #            counter = 0
        #    if counter == 1:
        #        temp.append(indices2)
        #    H[indices2[0],indices2[1]] = 0
        count = 1
    print(temp)
    return temp


# For understanding the concept of plotting the lines, the opencv was referred
# however, none of opencv functions were used.
def plot_lines(img2, max_index, ph, theta):

    for i in range(len(max_index)):

        dist = ph[max_index[i][0]]

        angle = theta[max_index[i][1]]
        idy = np.sin(angle)
        idx = np.cos(angle)

        y0 = idy * dist
        x0 = idx * dist

        y1 = int(y0 + 900 * idx)
        x1 = int(x0 + 900 * (-idy))

        y2 = int(y0 - 900 * idx)
        x2 = int(x0 - 900 * (-idy))

        cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

edges = sobel_operator()
# The border lines of the image are made 0 to avoid false positives.
for i in range(edges.shape[0]):
    for j in range(edges.shape[1]):
        if i <= 20:
            edges[i][j] = 0
        if i >= edges.shape[0] - 20:
            edges[i][j] = 0
        if j <= 20:
            edges[i][j] = 0
        if j >= edges.shape[1] - 20:
            edges[i][j] = 0


edges = segmentation(edges, 170)

def vertical_lines():
    # For vertical lines
    theta, ph, H = hough_transform(edges,90,180)
    # The number of peaks needs to be much higher than number of red lines to account for overlapping.
    max_index = peak_detection1(H,peaks=17)
    plot_lines(imgg,max_index,ph,theta)
    cv2.imwrite('red_line.jpg',imgg)


def diagonal_lines():
    # For diagonal lines
    theta, ph, H = hough_transform(edges,75,150)
    print('Reached here')
    # The number of peaks needs to be much higher than number of red lines to account for overlapping.
    max_index = peak_detection1(H,peaks=50)
    imgg = cv2.imread('hough.jpg')
    plot_lines(imgg,max_index,ph,theta)
    cv2.imwrite('blue_lines.jpg',imgg)

vertical_lines()
diagonal_lines()
# theta,radius,H = hough_transform_circle(edges)


