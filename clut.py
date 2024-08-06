import filecmp
import glob
import struct

import cv2
import numba
import numpy as np
from matplotlib import pyplot as plt

CLUT_DIM_SIZE = 17
LARGE_ARRAY_NUM = 729
SMALL_ARRAY_NUM = 648


def set_to_int(b, g, r):
    d = (b << 20) | (g << 10) | r
    return d


def decimal_to_signed_binary(n):
    if n >= 0:
        temp = format(n, '010b')  # 正数的二进制表示
    else:
        # 负数的二进制表示：取反并加1
        temp = format((1 << 10) + n, '010b')
    # value = 0
    # for i in range(len(temp)):
    #     if temp[i] == '1':
    #         value |= (1 << (9 - i))
    return int(temp, 2)


def compress_old(lut):
    c0 = np.zeros(LARGE_ARRAY_NUM)
    c1 = np.zeros(LARGE_ARRAY_NUM)
    c2 = np.zeros(LARGE_ARRAY_NUM)
    c3 = np.zeros(LARGE_ARRAY_NUM)

    c4 = np.zeros(SMALL_ARRAY_NUM)
    c5 = np.zeros(SMALL_ARRAY_NUM)
    c6 = np.zeros(SMALL_ARRAY_NUM)
    c7 = np.zeros(SMALL_ARRAY_NUM)

    index_c0 = index_c1 = index_c2 = index_c3 = index_c4 = index_c5 = index_c6 = index_c7 = 0

    for k in range(CLUT_DIM_SIZE):
        for j in range(CLUT_DIM_SIZE):
            for i in range(CLUT_DIM_SIZE):
                index = (i & 1) + ((j & 1) << 1) + ((k & 1) << 2)
                r = (lut[0][i][j][k] * 1023).astype(int)
                g = (lut[1][i][j][k] * 1023).astype(int)
                b = (lut[2][i][j][k] * 1023).astype(int)

                r = decimal_to_signed_binary(r)
                g = decimal_to_signed_binary(g)
                b = decimal_to_signed_binary(b)

                rgb = set_to_int(b, g, r)

                if rgb < 0:
                    print('error value')

                if index == 0:
                    c0[index_c0] = rgb
                    index_c0 = index_c0 + 1
                elif index == 1:
                    c1[index_c1] = rgb
                    index_c1 = index_c1 + 1
                elif index == 2:
                    c2[index_c2] = rgb
                    index_c2 = index_c2 + 1
                elif index == 3:
                    c3[index_c3] = rgb
                    index_c3 = index_c3 + 1
                elif index == 4:
                    c4[index_c4] = rgb
                    index_c4 = index_c4 + 1
                elif index == 5:
                    c5[index_c5] = rgb
                    index_c5 = index_c5 + 1
                elif index == 6:
                    c6[index_c6] = rgb
                    index_c6 = index_c6 + 1
                elif index == 7:
                    c7[index_c7] = rgb
                    index_c7 = index_c7 + 1

    all_data = np.concatenate((c0, c4, c1, c5, c2, c6, c3, c7)).astype(int).reshape(4, 1377)
    all_data = all_data.T
    all_data = all_data.reshape(1, -1)

    with open(result_name1, 'w') as file:
        for num in all_data[0]:
            file.write(str(num) + ',')


def compress_old1(lut):
    c0 = np.zeros(LARGE_ARRAY_NUM)
    c1 = np.zeros(LARGE_ARRAY_NUM)
    c2 = np.zeros(LARGE_ARRAY_NUM)
    c3 = np.zeros(LARGE_ARRAY_NUM)

    c4 = np.zeros(SMALL_ARRAY_NUM)
    c5 = np.zeros(SMALL_ARRAY_NUM)
    c6 = np.zeros(SMALL_ARRAY_NUM)
    c7 = np.zeros(SMALL_ARRAY_NUM)

    all_data = np.zeros(5508).astype(int)
    print(all_data.shape)
    index_c0 = index_c1 = index_c2 = index_c3 = index_c4 = index_c5 = index_c6 = index_c7 = 0

    for k in range(CLUT_DIM_SIZE):
        for j in range(CLUT_DIM_SIZE):
            for i in range(CLUT_DIM_SIZE):
                index = (i & 1) + ((j & 1) << 1) + ((k & 1) << 2)
                r = (lut[0][j][k][i] * 1023).astype(int)
                g = (lut[1][j][k][i] * 1023).astype(int)
                b = (lut[2][j][k][i] * 1023).astype(int)

                r = decimal_to_signed_binary(r)
                g = decimal_to_signed_binary(g)
                b = decimal_to_signed_binary(b)

                rgb = set_to_int(b, g, r)

                if rgb < 0:
                    print('error value')

                if index == 0:
                    index = index_c0 * 4 + index % 4
                    all_data[index] = rgb
                    index_c0 = index_c0 + 1
                elif index == 1:
                    index = index_c1 * 4 + index % 4
                    all_data[index] = rgb
                    index_c1 = index_c1 + 1
                elif index == 2:
                    index = index_c2 * 4 + index % 4
                    all_data[index] = rgb
                    index_c2 = index_c2 + 1
                elif index == 3:
                    index = index_c3 * 4 + index % 4
                    all_data[index] = rgb
                    index_c3 = index_c3 + 1
                elif index == 4:
                    index = (LARGE_ARRAY_NUM + index_c4) * 4 + index % 4
                    all_data[index] = rgb
                    index_c4 = index_c4 + 1
                elif index == 5:
                    index = (LARGE_ARRAY_NUM + index_c5) * 4 + index % 4
                    all_data[index] = rgb
                    index_c5 = index_c5 + 1
                elif index == 6:
                    index = (LARGE_ARRAY_NUM + index_c6) * 4 + index % 4
                    all_data[index] = rgb
                    index_c6 = index_c6 + 1
                elif index == 7:
                    index = (LARGE_ARRAY_NUM + index_c7) * 4 + index % 4
                    all_data[index] = rgb
                    index_c7 = index_c7 + 1
    all_data = all_data.astype(int)

    with open(result_name2, 'w') as file:
        file.write(','.join(map(str, all_data)) + '\n')


def compress(lut):
    c = [np.zeros(LARGE_ARRAY_NUM) for _ in range(4)] + [np.zeros(SMALL_ARRAY_NUM) for _ in range(4)]
    indices = [0] * 8

    for k in range(CLUT_DIM_SIZE):
        for j in range(CLUT_DIM_SIZE):
            for i in range(CLUT_DIM_SIZE):
                index = (k & 1) + ((i & 1) << 1) + ((j & 1) << 2)
                rgb = (lut[:, k, i, j] * 511).astype(int)
                c[index][indices[index]] = set_to_int(rgb[2], rgb[1], rgb[0]).astype(int)
                indices[index] += 1

    all_data = np.concatenate(c).astype(int)

    data = struct.pack('5508i', *all_data)
    with open(result_name2, 'wb') as f:
        f.write(data)


@numba.jit
def modf(b, binsize):
    d = b / float(binsize)
    quotient = np.floor(d)
    remainder = d - quotient
    return remainder


@numba.jit
def trilinear_cpu(lut, image, width, height, binsize):
    r_img = image[:, :, 0]
    g_img = image[:, :, 1]
    b_img = image[:, :, 2]

    lut_r = lut[0, :, :, :]
    lut_g = lut[1, :, :, :]
    lut_b = lut[2, :, :, :]

    out_img = np.empty_like(image)

    for y in range(height):
        for x in range(width):
            r = r_img[y, x]
            g = g_img[y, x]
            b = b_img[y, x]

            k = r_id = int(np.floor(r / binsize))
            j = g_id = int(np.floor(g / binsize))
            i = b_id = int(np.floor(b / binsize))
            # r_d = np.modf(r / binsize)[0]
            # g_d = np.modf(g / binsize)[0]
            # b_d = np.modf(b / binsize)[0]
            r_d = modf(r, binsize)
            g_d = modf(g, binsize)
            b_d = modf(b, binsize)

            w000 = (1 - r_d) * (1 - g_d) * (1 - b_d)
            w100 = r_d * (1 - g_d) * (1 - b_d)
            w010 = (1 - r_d) * g_d * (1 - b_d)
            w110 = r_d * g_d * (1 - b_d)
            w001 = (1 - r_d) * (1 - g_d) * b_d
            w101 = r_d * (1 - g_d) * b_d
            w011 = (1 - r_d) * g_d * b_d
            w111 = r_d * g_d * b_d

            # w1 = w000 * lut_r[i, j, k]
            # w2 = w100 * lut_r[i, j, k + 1]
            # w3 = w010 * lut_r[i, j + 1, k]
            # w4 = w110 * lut_r[i, j + 1, k + 1]
            # w5 = w001 * lut_r[i + 1, j, k]
            # w6 = w101 * lut_r[i + 1, j, k + 1]
            # w7 = w011 * lut_r[i + 1, j + 1, k]
            # w8 = w111 * lut_r[i + 1, j + 1, k + 1]
            #
            # w = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8

            out_img[y, x, 0] = (w000 * lut_r[i, j, k] + w100 * lut_r[i, j, k + 1] + \
                                w010 * lut_r[i, j + 1, k] + w110 * lut_r[i, j + 1, k + 1] + \
                                w001 * lut_r[i + 1, j, k] + w101 * lut_r[i + 1, j, k + 1] + \
                                w011 * lut_r[i + 1, j + 1, k] + w111 * lut_r[i + 1, j + 1, k + 1]) * 127

            out_img[y, x, 1] = (w000 * lut_g[i, j, k] + w100 * lut_g[i, j, k + 1] + \
                                w010 * lut_g[i, j + 1, k] + w110 * lut_g[i, j + 1, k + 1] + \
                                w001 * lut_g[i + 1, j, k] + w101 * lut_g[i + 1, j, k + 1] + \
                                w011 * lut_g[i + 1, j + 1, k] + w111 * lut_g[i + 1, j + 1, k + 1]) * 127

            out_img[y, x, 2] = (w000 * lut_b[i, j, k] + w100 * lut_b[i, j, k + 1] + \
                                w010 * lut_b[i, j + 1, k] + w110 * lut_b[i, j + 1, k + 1] + \
                                w001 * lut_b[i + 1, j, k] + w101 * lut_b[i + 1, j, k + 1] + \
                                w011 * lut_b[i + 1, j + 1, k] + w111 * lut_b[i + 1, j + 1, k + 1]) * 127
    return out_img.astype(np.uint8)


if __name__ == '__main__':
    dir_path = "bin/"
    bin = glob.glob(dir_path + "raw.bin")[0]

    src_img = glob.glob(dir_path + "1.jpg")[0]

    result_name1 = 'bin/result1.txt'
    result_name2 = 'bin/result2.txt'

    # 从二进制文件读取数据，指定小端模式
    lut = np.fromfile(bin, dtype='<f4')
    lut = lut.reshape((3, 17, 17, 17))
    lut_transposed = np.transpose(lut, (0, 3, 1, 2))

    compress_old(lut_transposed)
    compress_old1(lut)

    # result = filecmp.cmp(result_name1, result_name2)
    # if result:
    #     print("The files are identical")
    # else:
    #     print("The files are different")

    with open(result_name1, 'r') as file1:
        with open(result_name2, 'r') as file2:
            # 读取文件内容
            content1 = file1.read()
            content2 = file2.read()

            # 比较两个文件的内容
            if content1 == content2:
                print("文件内容相同")
            else:
                # 找出不同的位置
                for i, (c1, c2) in enumerate(zip(content1, content2)):
                    if c1 != c2:
                        print(f"文件内容在位置 {i} 处不同：{c1} vs {c2}")

    img = cv2.imread(src_img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # plt.imshow(img_rgb)
    # plt.title("input")
    # plt.show()
    
    print(img_rgb.shape)
    print(lut.shape)
    #lut = lut * 2
    out_img = trilinear_cpu(lut, img_rgb, img_rgb.shape[1], img_rgb.shape[0], 16)
    
    # plt.imshow(out_img)
    # plt.title("out img")
    # plt.show()
   # out_img = out_img*2
    fake_img = img_rgb + out_img
    
    # plt.imshow(fake_img)
    # plt.title("fake_img")
    # plt.show()
    
    out_bgr = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('bin/pc_out1.jpg', out_bgr)















    # compress_old1(lut)
    # compress(lut_transposed)


