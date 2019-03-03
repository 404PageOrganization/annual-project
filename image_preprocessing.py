# -*- coding: utf-8 -*-

from PIL import Image, ImageChops
import os
import numpy as np
import cv2


in_dir = "real_img_origin/"
out_dir = "real_img/"
val_min = 10    # the threshold value
range_min = 60    # the minimum size of a character
count = 0    # the serial number


def extract_peek(array_vals, val_min, range_min):
    # extract the peek of the projection
    i_start = None
    i_end = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > val_min and i_start is None:
            i_start = i
        elif val < val_min and i_start is not None:
            if i - i_start >= range_min:
                i_end = i
                peek_ranges.append((i_start, i_end))
                i_start = None
                i_end = None
    return peek_ranges


def crop_char(img, peek_range):
    #
    global count
    for i, peek_range in enumerate(peek_ranges):
        for vertical_range in vertical_peek_ranges2d[i]:
            x = vertical_range[0]
            y = peek_range[0]
            w = vertical_range[1] - x
            h = peek_range[1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            count += 1
            img1 = img[y:peek_range[1], x:vertical_range[1]]
            crop_blank(img1)


def crop_blank(img):
    # remove the blank edges
    img = Image.fromarray(img)
    pix = img.load()
    print(pix[0, 0])
    w, h = img.size
    x_min = w
    y_min = h
    x_max = -1
    y_max = -1
    for x in range(0, w - 1):
        for y in range(0, h - 1):
            if pix[x, y] != 255:
                if x_min > x:
                    x_min = x
                if y_min > y:
                    y_min = y
                if x_max < x:
                    x_max = x
                if y_max < y:
                    y_max = y
    img = img.crop([x_min, y_min, x_max, y_max])
    resize_img(img)


def resize_img(img):
    # resize the image into a 128*128 one and output
    w, h = img.size
    if w == h:
        w1 = 80
        h1 = 80
    elif w < h:
        w1 = int(w * 80 / h)
        h1 = 80
    elif w > h:
        w1 = 80
        h1 = int(h * 80 / w)
    img = img.resize((w1, h1), Image.ANTIALIAS)
    img_new = Image.new(mode='L', size=(128, 128), color=255)
    img_new.paste(img, (int(64 - w1 / 2), int(64 - h1 / 2) + 12))
    img_new.save(out_dir + str(count) + ".png", 'png')


print('strat running')

for file_name in os.listdir(in_dir):
    # read raw image
    img = cv2.imread(in_dir + file_name, 0)
    print('succeeded: reading image')

    # binaryzation & denoising
    ret, img_bi = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    # img_bi = cv2.adaptiveThreshold(img, 255,
    #                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv2.THRESH_BINARY_INV, 11, 2)
    img_dst = img_bi
    img_dst = cv2.fastNlMeansDenoising(img_bi, 10, 7, 21)
    cv2.imwrite(out_dir + '!' + file_name, img_dst)
    print('succeeded: binaryzing & denoising')

    # horizontal projection
    horizontal_sum = np.sum(img_dst, axis=1)
    peek_ranges = extract_peek(horizontal_sum, val_min, range_min)
    line_seg_img_dst = np.copy(img_dst)
    for i, peek_range in enumerate(peek_ranges):
        x = 0
        y = peek_range[0]
        w = line_seg_img_dst.shape[1]
        h = peek_range[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(line_seg_img_dst, pt1, pt2, 255)
    # vertical projection
    vertical_peek_ranges2d = []
    for peek_range in peek_ranges:
        start_y = peek_range[0]
        end_y = peek_range[1]
        line_img = img_dst[start_y:end_y, :]
        vertical_sum = np.sum(line_img, axis=0)
        vertical_peek_ranges = extract_peek(vertical_sum, val_min, range_min)
        vertical_peek_ranges2d.append(vertical_peek_ranges)
    print('succeeded: projecting in two directions')

    # invert the image into a normal one
    #img2 = img.copy()
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] > 180:
                img[i, j] = 255
            #img2[i,j]= 255 - img[i,j]
    #img = Image.fromarray(img)
    #img2 = Image.fromarray(img2)
    #img = Image.merge('LA', (img2, img))
    crop_char(img, peek_range)
