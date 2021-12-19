class Edge_Detector:
    def __init__(self, px_arr, kernel):
        self.conv_kernel = [1, 0, -1]  # Derivative filter. Is flipped for convolution

        # kernel/mask is assumed to be (m x m) size, where m is and odd number.
        self.kernel = kernel  # 'my_kernel' in main.py
        self.avg = [[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]]
        # averaging filter for smoothing.
        # 1/9 is multiplied later.

        self.image = px_arr  # Input image array

    # Gradient - Derivative operator
    def convolution_1d(self, img_in=None):
        if img_in is not None:  # when input image array to use is specified.
            horizontal_result = img_in.copy()
            vertical_result = img_in.copy()
            Magnitude = img_in.copy()
            # Get a copy of img array to save convolution result.
            image = img_in
        else:  # when input image array is not specified, use copy of self.image.
            horizontal_result = self.image.copy()
            vertical_result = self.image.copy()
            Magnitude = self.image.copy()
            image = self.image

        for i in range(len(image)):  # i for row
            for j in range(len(image[0])):  # j for column
                for k in range(3):  # repeating for  R, G, B values.
                    # horizontal convolution
                    if j == 0:  # 필터가 이미지 영역 밖으로 나갈 때. (필터 좌측)
                        horizontal_result[i][j][k] = abs(image[i][j + 1][k] * self.conv_kernel[2])
                        # 이미지 영역 안에 있는 부분만 계산. (0 패딩을 가정)
                    elif j == len(image[0]) - 1:  # 필터가 이미지 영역 밖으로 나갈 때. (필터 우측)
                        horizontal_result[i][j][k] = abs(image[i][j - 1][k] * self.conv_kernel[0])
                        # 이미지 영역 안에 있는 부분만 계산. (0 패딩을 가정)
                    else:
                        try:
                            horizontal_result[i][j][k] = abs(image[i][j - 1][k] * self.conv_kernel[0] +
                                                             image[i][j][k] * self.conv_kernel[1] +
                                                             image[i][j + 1][k] * self.conv_kernel[2])
                            # 각 이미지 픽셀 값과 동일 위치의 필터 값을 곱하여 합을 구하고 저장.
                        except IndexError:
                            print(i, j)  # 디버깅 용도
                            return

                    # vertical convolution
                    if i == 0:  # 필터가 이미지 영역 밖으로 나갈 때. (필터 상단)
                        vertical_result[i][j][k] = abs(self.image[i + 1][j][k] * self.conv_kernel[2])
                        # 이미지 영역 안에 있는 부분만 계산. (0 패딩을 가정)
                    elif i == len(self.image) - 1:  # 필터가 이미 영역 밖으로 나갈 때. (필터 하단)
                        vertical_result[i][j][k] = abs(self.image[i - 1][j][k] * self.conv_kernel[0])
                        # 이미지 영역 안에 있는 부분만 계산. (0 패딩을 가정)
                    else:
                        try:
                            vertical_result[i][j][k] = abs(self.image[i - 1][j][k] * self.conv_kernel[0] +
                                                           self.image[i][j][k] * self.conv_kernel[1] +
                                                           self.image[i + 1][j][k] * self.conv_kernel[2])
                            # 각 이미지 픽셀 값과 동일 위치의 필터 값을 곱하여 합을 구하고 저장.
                        except IndexError:
                            print(i, j)  # 디버깅 용도
                            return
                    # Magnitude
                    Magnitude[i][j][k] = (horizontal_result[i][j][k] ** 2 + vertical_result[i][j][k] ** 2) ** 0.5
                    # horizontal, vertical 값을 제곱하여 더한 것의 제곱근을 구하여 크기(Magnitude)를 찾고 저장.

        return horizontal_result, vertical_result, Magnitude  # image array 들을 반환

    # 2D Convolution implementing my own kernel
    # -----------------------------------------------------------------------------------------------------------
    def convolution_2d(self, mask_in=None, img_in=None, divisor=1):
        if mask_in is not None:
            mask = mask_in
        else:
            mask = self.kernel  # 입력 받은 mask_in 값이 없으면 self.kernel = my_kernel 을 기본으로 사용.
        if img_in is not None:
            img_in = img_in
            convolution_img = img_in.copy()  # Get a copy of img array to save convolution result.
        else:
            img_in = self.image  # when input image array is not specified, use copy of self.image.
            convolution_img = self.image.copy()

        size = len(mask)  # 이미지에서 컨볼루션 연산을 수행할 부분의 크기 = 마스크의 크기
        for row in range(len(img_in)):
            for col in range(len(img_in[0])):
                # for each pixel in image
                area = self.setArea(size, row, col, img_in)  # 컨볼루션 연산을 수행할 이미지 영역을 구한다. setArea 함수는 아래에 존재.
                convolution_img[row][col] = self.img_conv(mask, area, divisor=divisor)
                # mask(filter)와 이미지 영역 일부분과 컨볼루션 연산 수행하고 그 값을 저장.
        return convolution_img  # image array 반환

    def smoothing_conv(self, smoothing_filter=None, kernel=None):
        if kernel is not None:  # after_kernel: smoothing 이후 컨볼루션 연산에 사용할 커널(필터)
            after_kernel = kernel
        else:
            after_kernel = self.kernel

        if smoothing_filter is not None:
            smooth_img = self.convolution_2d(mask_in=smoothing_filter, img_in=self.image, divisor=1)
            # smoothing filter 와 image array 의 컨볼루션 연산을 수행.

        else:  # smoothing filter 가 명시되지 않았을 경우, averaging filter 를 default 로 사용
            smooth_img = self.convolution_2d(mask_in=self.avg, img_in=self.image, divisor=len(after_kernel) ** 2)
            # smoothing filter 와 image array 의 컨볼루션 연산을 수행. (averaging)

        convolution_img = self.convolution_2d(mask_in=after_kernel, img_in=smooth_img)
        # smoothing 과정을 거친 image 와 after_kernel 간 컨볼루션 연산을 수행.
        return convolution_img  # image array 반환

    # General Functions
    # -----------------------------------------------------------------------------------------------------------
    def setArea(self, size, row, col, img):  # 컨볼루션을 수행할 영역을 구하는 함수.
        area = [[[0, 0, 0] for i in range(size)] for j in range(size)]  # (size x size) 크기의 영역을 초기화
        for h in range(size):  # h for horizontal (row)
            for v in range(size):  # v for vertical (column)
                try:
                    # 영역의 (h,v)인덱스 위치가 이미지 바깥으로 나가서 매핑되는 값이 없는 경우, 0으로 패딩.
                    if row - int(size / 2) + h < 0 or col - int(size / 2) + v < 0:
                        area[h][v] = [0, 0, 0]
                    elif row - int(size / 2) + h >= len(img) or col - int(size / 2) + v >= len(img[0]):
                        area[h][v] = [0, 0, 0]
                    else:
                        # 이미지에 매핑되는 인덱스의 픽셀 값을 영역에 저장.
                        area[h][v] = img[row - int(size / 2) + h][col - int(size / 2) + v]

                except IndexError:  # 디버깅 용
                    print("h: ", h, "v: ", v)
                    print(row - int(size / 2) + h, col - int(size / 2) + v)
                    break
        return area  # 구한 (size x size) 크기의 영역을 반환.

    def img_conv(self, mask, area, divisor=1):  # mask(filter)와 image array 간의 컨볼루션 연산을 수행.
        # divisor: smoothing 에서 averaging filter 사용시 나중에 공통으로 나눠주는 값
        sum_r, sum_g, sum_b = 0, 0, 0  # RGB 값 초기화
        for row in range(len(mask)):
            for col in range(len(mask[0])):
                # 각 이미지 픽셀 RGB 값과 동일 위치의 필터 값을 곱하여 합을 구하고 저장.
                sum_r += mask[row][col] * area[row][col][0]
                sum_g += mask[row][col] * area[row][col][1]
                sum_b += mask[row][col] * area[row][col][2]
        return [abs(sum_r) / divisor, abs(sum_g) / divisor, abs(sum_b) / divisor]
        # [R, G, B] 값을 반환
