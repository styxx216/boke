import numpy as np
import cv2
from numba import jit


@jit()
def add_pixels(image, gor=1, vert=1):
    res = image
    n = image.shape[0]
    m = image.shape[1]
    add = np.zeros([n, gor, 3])
    res = np.hstack([add, res])
    res = np.hstack([res, add])
    add = np.zeros([vert, m + 2 * gor, 3])
    res = np.vstack([add, res])
    res = np.vstack([res, add])
    return res


@jit()
def bokeh(image, core):
    image = image / 255
    core = core / 255
    res = image
    n = core.shape[0]
    m = core.shape[1]
    N = res.shape[0]
    M = res.shape[1]
    image_with_pixels = add_pixels(image, m // 2, n // 2)
    sum = np.sum(core)
    matrix = np.zeros([n, m])
    for i in range(N):
        for j in range(M):
            for k in range(3):
                for x in range(n):
                    for y in range(m):
                        matrix[x, y] = core[x, y] * image_with_pixels[i + x, j + y, k]
                res[i, j, k] = np.sum(matrix) / sum
    return res * 255


def main(path_image='J:\\1\\1\\kWyZaAsis8.jpg',
         path_core='J:\\1\\1\\core23.png',
         path_final_image='J:\\1\\1\\kWyZaAsis8+core23.jpg'):
    image = cv2.imread(path_image)
    core = cv2.imread(path_core, cv2.IMREAD_GRAYSCALE)
    final_image = bokeh(image, core)
    cv2.imwrite(path_final_image, final_image)


main()
