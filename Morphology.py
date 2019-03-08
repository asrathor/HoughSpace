import numpy as np
import cv2
from copy import deepcopy

# Load the image
img1 = cv2.imread('noise.jpg',0)
# Normalize the image by 255
img1 = img1/255

# Both padding_new and padding function is used to pad the image at different sizes based
# on the size of the kernel. These functions are taken from my project 1 submission.
def padding_new(matrixx):

    final_mat = np.zeros((matrixx.shape[0] + 4, matrixx.shape[1] + 4))

    final_mat[2:-2, 2:-2] = matrixx

    return final_mat

# Read description above.
def padding(matrixx):

    final_mat = np.zeros((matrixx.shape[0] + 2, matrixx.shape[1] + 2))

    for i in range(matrixx.shape[0]):
        for j in range(matrixx.shape[1]):
            final_mat[i + 1][j + 1] = matrixx[i][j]

    return final_mat

# Creates a kernel of size 5,5 filled with ones.
def struct_kernel():

    kernel = np.ones((5,5),np.uint8)
    return kernel

# This function slices the matrix in size 5,5.
def slice_mat(matrixx,index_x,index_y):

    slice_size = 5
    mat = np.zeros((slice_size, slice_size))

    for i in range(slice_size):
        for j in range(slice_size):

            mat[i][j] = matrixx[index_x+i][index_y+j]

    return mat

# This function slices the matrix in size 3,3.
def slice_mat1(matrixx,index_x,index_y):

    slice_size = 3
    mat = np.zeros((slice_size, slice_size))

    for i in range(slice_size):
        for j in range(slice_size):

            mat[i][j] = matrixx[index_x+i][index_y+j]

    return mat

# This function compares whether two matrixes are identical. Used for erosion.
def compare_mat(matrix1,matrix2):

    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            if matrix1[i][j] != matrix2[i][j]:
                return False

    return True

# This function performs erosion but for kernel size 3,3. Used for boundary extraction.
def erosion1(img, kernel1, counter):

    print(img.shape)
    padded_img = padding(img)
    print(padded_img.shape)
    if counter == 0:
        new_img = np.zeros((img.shape[0],img.shape[1]))
    else:
        new_img = deepcopy(img)

    count = 0
    for i in range(padded_img.shape[0]-2):
        for j in range(padded_img.shape[1]-2):
            mats = slice_mat1(padded_img,i,j)
            if compare_mat(mats,kernel1) == True:
                count = count + 1
                new_img[i][j] = 1
            else:
                new_img[i][j] = 0

    return new_img

# This function performs erosion for kernel size 5,5.
def erosion(img, kernel, counter):

    print(img.shape)
    padded_img = padding_new(img)
    print(padded_img.shape)
    if counter == 0:
        new_img = np.zeros((img.shape[0],img.shape[1]))
    else:
        new_img = deepcopy(img)

    count = 0
    for i in range(padded_img.shape[0]-4):
        for j in range(padded_img.shape[1]-4):
            mats = slice_mat(padded_img,i,j)
            if compare_mat(mats,kernel) == True:
                count = count + 1
                new_img[i][j] = 1
            else:
                new_img[i][j] = 0

    return new_img

# This function is used to compare whether any values in two matrix matches with each
# other. Used for dilation.
def compare_mat2(matrix1,matrix2):

    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            if matrix1[i][j] == matrix2[i][j]:
                return True

    return False

# This function is used to perform dilation.
def dilation(new_img,kernel,counter):

    padded_img = padding_new(new_img)
    if counter == 0:
        temp = deepcopy(new_img)
    else:
        temp = np.zeros((new_img.shape[0],new_img.shape[1]))

    for i in range(padded_img.shape[0]-4):
        for j in range(padded_img.shape[1]-4):
            mats = slice_mat(padded_img,i,j)
            if compare_mat2(mats,kernel) == True:
                temp[i][j] = 1
            else:
                temp[i][j] = 0
    return temp


kernel = struct_kernel()

# This function is used to perform opening using previously implemented erosion and dilation.
def open_morph():
    # open
    eroded_img = erosion(img1,kernel,0)
    dilated_img = dilation(eroded_img,kernel,0)
    sec_dilated_img = dilation(dilated_img,kernel,0)
    sec_eroded_img = erosion(sec_dilated_img,kernel,1)
    sec_eroded_img = sec_eroded_img * 255
    cv2.imwrite('res_noise1.jpg',sec_eroded_img)
    sec_eroded_img = sec_eroded_img/255
    return sec_eroded_img

# This function is used to perform closing using previously implemented erosion and dilation.
def close_morph():
    # close
    dilated_img = dilation(img1,kernel,1)
    eroded_img = erosion(dilated_img,kernel,1)
    sec_eroded_img = erosion(eroded_img,kernel,1)
    sec_dilated_img = dilation(sec_eroded_img,kernel,0)
    sec_dilated_img = sec_dilated_img * 255
    cv2.imwrite('res_noise2.jpg',sec_dilated_img)
    sec_dilated_img = sec_dilated_img/255
    return sec_dilated_img

# This function is used to perform boundary extraction using erosion.
def boundary_ext(img2,counter):

    kernel1 = np.ones((3, 3), np.uint8)
    eroded_img = erosion1(img2,kernel1,0)
    boundary = img2 - eroded_img
    if counter == 0:
        boundary = boundary * 255
        cv2.imwrite('res_bound1.jpg',boundary)
    else:
        boundary = boundary * 255
        cv2.imwrite('res_bound2.jpg',boundary)


boundary_ext(open_morph(),0)
boundary_ext(close_morph(),1)
