import numpy as np


class Edge_Detector:
    def __init__(self, px_arr, kernel):
        self.conv_kernel = [1, 0, -1]
        self.corr_kernel = [-1, 0, 1]

        # kernel/mask is assumed to be mxm size, where m is and odd number.
        self.kernel = kernel
        self.avg = [[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]]
        # 1/9 is multiplied later.

        self.image = px_arr

    # Gradient - Derivative operator
    def convolution_1d(self, img_in=None):
        if img_in is not None:
            horizontal_result = img_in.copy()
            vertical_result = img_in.copy()
            Magnitude = img_in.copy()
            image = img_in
        else:
            horizontal_result = self.image.copy()
            vertical_result = self.image.copy()
            Magnitude = self.image.copy()
            image = self.image

        for i in range(len(image)):
            for j in range(len(image[0])):
                for k in range(3):
                    # horizontal convolution
                    if j == 0:
                        horizontal_result[i][j][k] = abs(image[i][j + 1][k] * self.conv_kernel[2])
                    elif j == len(image[0]) - 1:
                        horizontal_result[i][j][k] = abs(image[i][j - 1][k] * self.conv_kernel[0])
                    else:
                        try:
                            horizontal_result[i][j][k] = abs(image[i][j - 1][k] * self.conv_kernel[0] +
                                                             image[i][j][k] * self.conv_kernel[1] +
                                                             image[i][j + 1][k] * self.conv_kernel[2])
                        except IndexError:
                            print(i, j)
                            return
                    # vertical convolution
                    if i == 0:
                        vertical_result[i][j][k] = abs(self.image[i + 1][j][k] * self.conv_kernel[2])
                    elif i == len(self.image) - 1:
                        vertical_result[i][j][k] = abs(self.image[i - 1][j][k] * self.conv_kernel[0])
                    else:
                        try:
                            vertical_result[i][j][k] = abs(self.image[i - 1][j][k] * self.conv_kernel[0] +
                                                           self.image[i][j][k] * self.conv_kernel[1] +
                                                           self.image[i + 1][j][k] * self.conv_kernel[2])
                        except IndexError:
                            print(i, j)
                            return
                    # Magnitude
                    Magnitude[i][j][k] = (horizontal_result[i][j][k] ** 2 + vertical_result[i][j][k] ** 2) ** 0.5

        return horizontal_result, vertical_result, Magnitude

    # 2D Convolution implementing my own kernel
    # -----------------------------------------------------------------------------------------------------------
    def convolution_2d(self, mask_in=None, img_in=None, divisor=1):
        if mask_in is not None:
            mask = mask_in
        else:
            mask = self.kernel
        if img_in is not None:
            img_in = img_in
            convolution_img = img_in.copy()
        else:
            img_in = self.image
            convolution_img = self.image.copy()

        size = len(mask)
        for row in range(len(img_in)):
            for col in range(len(img_in[0])):
                # for each pixel in image
                area = self.setArea(size, row, col, img_in)
                convolution_img[row][col] = self.img_conv(mask, area, divisor=divisor)
        return convolution_img

    def smoothing_conv(self, smoothing_filter=None, kernel=None):
        if kernel is not None:
            smooth_kernel = kernel
        else:
            smooth_kernel = self.kernel

        if smoothing_filter is not None:
            smooth_img = self.convolution_2d(mask_in=smoothing_filter, img_in=self.image, divisor=1)
        else:
            smooth_img = self.convolution_2d(mask_in=self.avg, img_in=self.image, divisor=len(smooth_kernel) ** 2)

        convolution_img = self.convolution_2d(mask_in=smooth_kernel, img_in=smooth_img)
        return convolution_img

    # General Functions
    # -----------------------------------------------------------------------------------------------------------
    def setArea(self, size, row, col, img):
        area = [[[0, 0, 0] for i in range(size)] for j in range(size)]
        for h in range(size):
            for v in range(size):
                try:
                    if row - int(size / 2) + h < 0 or col - int(size / 2) + v < 0:
                        area[h][v] = [0, 0, 0]
                    elif row - int(size / 2) + h >= len(img) or col - int(size / 2) + v >= len(img[0]):
                        area[h][v] = [0, 0, 0]
                    else:
                        area[h][v] = img[row - int(size / 2) + h][col - int(size / 2) + v]
                except IndexError:
                    print("h: ", h, "v: ", v)
                    print(row - int(size / 2) + h, col - int(size / 2) + v)
                    break
        return area

    def img_conv(self, mask, area, divisor=1):
        # divisor is for later division like average denominator.
        sum_r, sum_g, sum_b = 0, 0, 0
        for col in range(len(mask)):
            for row in range(len(mask[0])):
                sum_r += mask[col][row] * area[col][row][0]
                sum_g += mask[col][row] * area[col][row][1]
                sum_b += mask[col][row] * area[col][row][2]
        return [abs(sum_r) / divisor, abs(sum_g) / divisor, abs(sum_b) / divisor]
