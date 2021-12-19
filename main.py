from PIL import Image  # 이미지 로딩과 output 저장을 위한 라이브러리. nmupy와 호환되어 numpy 배열 형태로 이미지를 로드할 수 있다.
import numpy as np  # 1. numpy 배열 형태로 이미지를 담기 위한 용도.  2. np.exp()를 통해 exponential e^x 계산을 하기위한 용도.
from math import pi  # 원주율 파이 값 계산을 위한 용도.

# Conv.py 에서 만든 클래스 import
from Conv import Edge_Detector


def get_gaussian(size, sigma):  # 가우시안 필터 생성 함수.
    gaussian = [[0 for i in range(size)] for j in range(size)]  # size x size 크기 가우시안 필터 초기화.
    mid = int(size/2)  # 센터 위치 산출
    for x in range(size):  # x for row
        for y in range(size):  # y for column
            gaussian[x][y] = np.exp(-((mid-x) ** 2 + (mid-y) ** 2) / (2 * sigma ** 2)) / (2 * pi * sigma ** 2)
            # 센터로부터의 위치에 따른 가우시안 값 계산식
    return gaussian


def derivative_of_gaussian(size, sigma):  # DoG 필터 생성 함수
    DoG = [[0 for i in range(size)] for j in range(size)]  # size x size 크기 필터 초기화.
    mid = int(size / 2)  # 센터 위치
    for x in range(size):  # x for row
        for y in range(size):  # y for column
            # 시도한 방법 1: x,y 편미분한 식에 대입하여 나온 두 값을 더한 값으로 필터 값으로 사용해본 식
            DoG[x][y] = (-(mid-x) * np.exp(-((mid-x) ** 2 + (mid-y) ** 2) / (2 * sigma ** 2)) / sigma ** 2) + \
                        (-(mid-y) * np.exp(-((mid-y) ** 2 + (mid-x) ** 2) / (2 * sigma ** 2)) / sigma ** 2)
            # 결과는 방법 1이 더 깔끔해 보이는 결과가 나옴.
            # 시도한 방법 2: x,y 편미분한 식에 대입하여 나온 두 값으로 Magnitude 값을 구하여 필터 값으로 사용해본 식
            """DoG_x = (-(mid - x) * np.exp(-((mid - x) ** 2 + (mid - y) ** 2) / (2 * sigma ** 2)) / sigma ** 2)
            DoG_y = (-(mid - y) * np.exp(-((mid - y) ** 2 + (mid - x) ** 2) / (2 * sigma ** 2)) / sigma ** 2)
            DoG[x][y] = (DoG_y**2 + DoG_x**2) ** 0.5"""
    return DoG


def laplacian_of_gaussian(size, sigma):  # LoG 필터 생성 함수
    LoG = [[0 for i in range(size)] for j in range(size)]  # size x size 크기 필터 초기화.
    mid = int(size / 2)  # 센터 위치
    for x in range(size):  # x for row
        for y in range(size):  # y for column
            # 시도한 방법 1: x,y 편미분한 식에 대입하여 나온 두 값으로 Magnitude 값을 구하여 필터 값으로 사용해본 식
            # sigma 값이 높을 때 방법1 결과가 나쁘지 않게 나온 것 처럼 보임.
            LoG_x = ((mid-x) ** 2 / sigma ** 4 - 1/sigma**2) * np.exp(-((mid-x) ** 2 + (mid-y) ** 2) / (2 * sigma ** 2))
            LoG_y = ((mid-y) ** 2 / sigma ** 4 - 1/sigma**2) * np.exp(-((mid-x) ** 2 + (mid-y) ** 2) / (2 * sigma ** 2))
            LoG[x][y] = (LoG_y**2 + LoG_x**2) ** 0.5

            # 시도한 방법 2: x,y 편미분한 식에 대입하여 나온 두 값을 더한 값으로 필터 값으로 사용해본 식
            # 방법2의 결과들은 평균적으로는 방법1보다는 덜 지저분해 보였음
            # LoG[x][y] = LoG_y + LoG_x
    return LoG


if __name__ == '__main__':
    img = Image.open("./images/benalia.jpg")  # 이미지 로드
    pixel_array = np.array(img)  # 로딩한 이미지를 rgb 값의 배열로 변환
    my_kernel = [[2, 1, 0],
                [1, 0, -1],
                [0, -1, -2]]  # [1, 0, -1] Derivative kernel 두 개를 더해본 kernel (horizontal 방향 + vertical 방향)

    laplacian = [[0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]]  # Finite Difference Laplacian filter introduced in lecture note.

    detector = Edge_Detector(pixel_array, my_kernel)  # Edge_Detector class 인스턴스 생성.

    # selecting operation. 원하는 operation 을 True 바꿔서 사용
    get_1d = False  # Derivative filter convolution: [1, 0, -1]
    get_2d = False  # Convolution using 2d filter. Default set to 'my_kernel'
    get_gs = False  # gaussian smoothing
    get_DoG = False  # derivative of gaussian
    get_lap = False  # laplacian
    get_LoG = False  # laplacian of gaussian
    get_LG = True   # Compute gaussian smoothing, then laplacian convolution.
    # ---------------------------------------------------------------
    if get_1d:
        """
        [1, 0, -1] derivative kernel 을 horizontally, vertically 적용하여 각각의 결과를 확인하고, 
        두 결과로 Magnitude 를 구한 결과를 확인한다.
        """
        result_h, result_v, result_m = detector.convolution_1d()
        # 클래스 메소드를 통해 계산된 horizontal, vertical, magnitude 값

        img_hor = Image.fromarray(result_h, 'RGB')  # 배열 형태로 넘어온 값을 RGB 이미지로 변환
        img_hor.save("./output/1d_convolution/horizontal_mask.png")  # 이미지를 해당 디렉토리에 저장
        # horizontal direction 으로 [1, 0, -1] 커널을 적용한 것

        img_ver = Image.fromarray(result_v, 'RGB')
        img_ver.save("./output/1d_convolution/vertical_mask.png")
        # vertical direction 으로 [1, 0, -1] 커널을 적용한 것

        img_mag = Image.fromarray(result_m, 'RGB')
        img_mag.save("./output/1d_convolution/magnitude.png")
        # horizontal, vertical 값을 이용해 구한 magnitude 로 나타낸 결과 이미지.

    # ----------------------------------------------------------------
    if get_2d:
        result1 = detector.convolution_2d()  # 2d 필터와 convolution 한 결과를 반환. parameter 가 없으면 my_kernel 이 적용됨.
        reconstructed1 = Image.fromarray(result1, 'RGB')  # 배열 형태로 넘어온 값을 RGB 이미지로 변환
        reconstructed1.save("./output/2d_convolution/2d_convolution.png")  # 이미지를 해당 디렉토리에 저장

        result2 = detector.smoothing_conv()  # 필터링 이전에 smoothing 과정을 거침. parameter 가 없으면 averaging 필터가 적용됨.
        reconstructed2 = Image.fromarray(result2, 'RGB')
        reconstructed2.save("./output/2d_convolution/smoothing/averaging.png")

    # ----------------------------------------------------------------
    size = 7
    if get_gs:
        for sigma in range(1, 7, 2):  # Repeat for sigma=1, sigma=3, sigma=5
            gaussian_smoothing_filter = get_gaussian(size=size, sigma=sigma)  # get gaussian filter
            result3 = detector.smoothing_conv(smoothing_filter=gaussian_smoothing_filter)
            # 필터링 이전에 가우시안 필터로 smoothing 과정을 거침.
            reconstructed3 = Image.fromarray(result3, 'RGB')  # 배열 형태로 넘어온 값을 RGB 이미지로 변환
            reconstructed3.save(f"./output/2d_convolution/smoothing/gaussian_{size}x{size}_sigma{sigma}.png")
            # 이미지를 해당 디렉토리에 저장

    # ----------------------------------------------------------------
    if get_DoG:
        for sigma in range(1, 7, 2):  # Repeat for sigma=1, sigma=3, sigma=5
            DoG = derivative_of_gaussian(size=size, sigma=sigma)  # get (derivative of gaussian) filter
            DoG_result = detector.convolution_2d(mask_in=DoG)  # DoG 필터로 이미지와 컨볼루션 계산을 함.
            DoG_reconstructed = Image.fromarray(DoG_result, 'RGB')  # 배열 형태로 넘어온 값을 RGB 이미지로 변환
            DoG_reconstructed.save(f"./output/2d_convolution/smoothing/DoG_{size}x{size}_sigma{sigma}.png")
            # 이미지를 해당 디렉토리에 저장

    # -----------------------------------------------------------------
    if get_lap:
        lap_result = detector.convolution_2d(mask_in=laplacian)  # Laplacian mask 롤 사용하여 컨볼루션 계산을 함
        lap_reconstructed = Image.fromarray(lap_result, 'RGB')   # 배열 형태로 넘어온 값을 RGB 이미지로 변환
        lap_reconstructed.save(f"./output/2d_convolution/laplacian.png")  # 이미지를 해당 디렉토리에 저장

    # -----------------------------------------------------------------
    if get_LoG:
        for sigma in range(1, 7, 2):  # Repeat for sigma=1, sigma=3, sigma=5
            LoG = laplacian_of_gaussian(size=size, sigma=sigma)  # get (laplacian of gaussian) filter
            LoG_result = detector.convolution_2d(mask_in=LoG)  # LoG 필터로 이미지와 컨볼루션 계산을 함.
            LoG_reconstructed = Image.fromarray(LoG_result, 'RGB')  # 배열 형태로 넘어온 값을 RGB 이미지로 변환
            LoG_reconstructed.save(f"./output/2d_convolution/smoothing/LoG_{size}x{size}_sigma{sigma}.png")
            # 이미지를 해당 디렉토리에 저장

    # -----------------------------------------------------------------
    if get_LG:
        for sigma in range(1, 7, 2):  # Repeat for sigma=1, sigma=3, sigma=5
            gaussian = get_gaussian(size=size, sigma=sigma)  # get gaussian filter
            laplacian_gauss_result = detector.smoothing_conv(smoothing_filter=gaussian, kernel=laplacian)
            # 가우시안 필터로 smoothing 과정을 거치고 laplacian filter 와 컨볼루션 계산을 함.
            LG_reconstructed = Image.fromarray(laplacian_gauss_result, 'RGB')  # 배열 형태로 넘어온 값을 RGB 이미지로 변환
            LG_reconstructed.save(f"./output/2d_convolution/smoothing/LG_{size}x{size}_sigma{sigma}.png")
            # 이미지를 해당 디렉토리에 저장
