import math
import numpy as np
import cv2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 312324247


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    kernel_length = len(k_size)  # length of the 1-D array as a kernel
    in_signal = np.pad(in_signal, (kernel_length - 1, kernel_length - 1), 'constant')  # pad - add zeroes to the edges
    in_signal_length = len(in_signal)  # length of the in_signal: 1-D array
    result = np.zeros(in_signal_length - kernel_length + 1)

    for i in range(in_signal_length - kernel_length + 1):  # Move it over the other vector
        result[i] = (in_signal[i:i + kernel_length] * k_size).sum()  # Multiply the values
    return result


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    kernel = np.flip(kernel)  # flip
    image_h, image_w = in_image.shape  # get height and width of image
    kernel_h, kernel_w = kernel.shape  # get height and width kernel
    image_padded = np.pad(in_image, (kernel_h // 2, kernel_w // 2),
                          'edge')  # pad - add location of the number to the edge and rows as well

    result = np.zeros((image_h, image_w))

    # Multiply each cell in the kernel with its parallel cell in the image matrix
    # Sum all the multiplicities and place the sum in the output matrix at (x,y)
    for x in range(image_h):
        for y in range(image_w):
            result[x, y] = (image_padded[x:x + kernel_h, y:y + kernel_w] * kernel).sum()
    return result


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale image
    :return: (directions, magnitude)
    """

    #  MagG = ||G||
    magGx = np.array([[0, 0, 0],
                      [-1, 0, 1],
                      [0, 0, 0]])

    magGy = magGx.transpose()  # transpose

    x_derivative = conv2D(in_image, magGx)  # derivative
    y_derivative = conv2D(in_image, magGy)  # derivative

    directions = np.rad2deg(np.arctan2(y_derivative, x_derivative))
    magnitude = np.sqrt(np.square(x_derivative) + np.square(y_derivative))

    return directions, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    assert (k_size % 2 == 1)  # check the size of the Gaussian’ kernelSize, should always be an odd number.

    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    return conv2D(in_image, gaussian_kernel(k_size, sigma))


def gaussian_kernel(size, sigma):
    """
    implementation of  Gaussian kernel
    function help for blurImage1
       Create the gaussian
      :param size
      :param sigma
      :return: kernel
       """

    mid = size // 2
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x, y = i - mid, j - mid
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return kernel


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    assert (k_size % 2 == 1)  # check the size of the Gaussian’ kernelSize, should always be an odd number.

    sigma = int(round(0.3 * ((k_size - 1) * 0.5 - 1) + 0.8))
    kernel = cv2.getGaussianKernel(k_size, sigma)  # getGaussianKernel function

    return cv2.filter2D(in_image, -1, kernel,
                        borderType=cv2.BORDER_REPLICATE)  # The border of the images should be padded same as in the ’Convolution’ section (cv.BORDERREPLICATE)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    laplacianMatrix = np.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]])

    #  ZeroCrossingSimple use a simple image like the ’codeMonkey’
    image = conv2D(img, laplacianMatrix)  # Laplacian
    result = np.zeros(image.shape)

    for i in range(image.shape[0] - (laplacianMatrix.shape[0] - 1)):
        for j in range(image.shape[1] - (laplacianMatrix.shape[1] - 1)):
            if image[i][j] == 0:  # check all his neighbors
                if (image[i][j - 1] < 0 and image[i][j + 1] > 0) or \
                        (image[i][j - 1] < 0 and image[i][j + 1] < 0) or \
                        (image[i - 1][j] < 0 and image[i + 1][j] > 0) or \
                        (image[i - 1][j] > 0 and image[i + 1][j] < 0):
                    result[i][j] = 255  # crossing
            if image[i][j] < 0:
                if (image[i][j - 1] > 0) or (image[i][j + 1] > 0) or (image[i - 1][j] > 0) or (image[i + 1][j] > 0):
                    result[i][j] = 255
    return result


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    #  ZeroCrossingLOG use the image ’boxMan’
    image = cv2.GaussianBlur(img, (5, 5), 0)
    return edgeDetectionZeroCrossingSimple(image)


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    image = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
    rows, cols = image.shape
    helpForCircles = {}
    detectedCircles = []
    points = []
    edges = []
    ths = 0.47  # thresh (0.47 of the pixels of the circle we detect)
    moves = 100


    for t in range(min_radius, max_radius + 1):
        for p in range(moves):
            angle = 2 * math.pi * p / moves
            x = int(t * math.cos(angle))  # calculate
            y = int(t * math.sin(angle))  # calculate
            points.append((x, y, t))



    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 255:
                edges.append((i, j))

    for edge1, edge2 in edges:
        for d1, d2, t in points:
            a = edge2 - d2
            b = edge1 - d1
            p = helpForCircles.get((a, b, t))
            if p is None:
                p = 0
            helpForCircles[(a, b, t)] = p + 1




    circlesSorted = sorted(helpForCircles.items(), key=lambda i: -i[1])
    for circle, p in circlesSorted:
        x, y, t = circle
        if p / moves >= ths and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in detectedCircles):
            detectedCircles.append((x, y, t))

    return detectedCircles


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    filtered_image_my = np.zeros_like(in_image)
    filtered_image_CV = cv2.bilateralFilter(in_image, k_size, sigma_space, sigma_color)
    k_size = int(k_size / 2)  # calculate
    in_image = cv2.copyMakeBorder(in_image, k_size, k_size, k_size, k_size,
                                  cv2.BORDER_REPLICATE, None, value=0)
    for x in range(k_size, in_image.shape[0] - k_size):
        for y in range(k_size, in_image.shape[1] - k_size):
            pivot_v = in_image[x, y]
            neighbor_hood = in_image[x - k_size:x + k_size + 1, y - k_size:y + k_size + 1]
            delta = neighbor_hood.astype(int) - pivot_v
            # code from the model class exercise (Denoising & Bilateral)
            diff_gau = np.exp(-np.power(delta, 2) / (2 * sigma_space))
            gaus_e = cv2.getGaussianKernel(2 * k_size + 1, sigma=sigma_color)
            gaus_e = gaus_e.dot(gaus_e.T)
            combo = gaus_e * diff_gau
            result = ((combo * neighbor_hood / combo.sum()).sum())

            filtered_image_my[x - k_size, y - k_size] = round(result)

    return filtered_image_CV, filtered_image_my
