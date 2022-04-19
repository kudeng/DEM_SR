from PIL import Image
import os

import skimage.metrics as metrics
import cv2
import numpy as np
import matplotlib.pyplot as plt


def eccAlign(im1, im2, termination):

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
     number_of_iterations,  termination)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1, im2, warp_matrix, warp_mode, criteria, None, 5)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return im2_aligned, warp_matrix


number_of_iterations = 100000
termination_eps = 0.1


def MinMax(a, im_min=-9418, im_max=10602):
    t = (a - im_min) / (im_max - im_min)
    # t = t * 65535 - 32768
    return t.astype('float32')


def MeanShift(origin, target):
    o_mean = origin.mean()
    t_mean = target.mean()
    return origin - (o_mean - t_mean)


def Range32(im_min=-18257, im_max=21563):
    gap = im_max - im_min
    def wrap(x):
        assert (x <= 1).all() and (x >= 0).all(), 'x must between 0 and 1'
        return (x * gap + im_min).astype('int16')
    return wrap

def to8(x):
    return (x * 255).astype('uint8')


def crop(edge):
    return lambda x: x[edge:-edge, edge:-edge]


def lap_match_lose(img, depth):
    img = cv2.Laplacian(img, cv2.CV_32F, ksize=7)
    depth = cv2.Laplacian(depth, cv2.CV_32F, ksize=3)
    bias = depth - img
    bias = np.sqrt(np.maximum(bias, 0))
    return np.mean(bias)


import imageio as iio

def gen_aligned(im1, im2, size, tag):
    im1 = np.array(im1.resize(size))
    im2 = np.array(im2.resize(size))
    im2 = MinMax(im2, -18257, 21563)
    if tag == 'lr':
        im1 = MinMax(im1, -9418, 10602)
        im1 = MeanShift(im1, im2)
    else:
        im1 = MinMax(im1, 0, 255)
    try:
        ali, warp = eccAlign(im2, im1, 0.01)
    except:
        ali = im1
    if tag == 'lr':
        # im1 = Range32(-18257, 21563)(crop(64)(im1))
        # im2 = Range32(-18257, 21563)(crop(64)(im2))
        # im1 = MeanShift(im1, im2)
        # ali = Range32(-18257, 21563)(crop(64)(ali))
        # ali = MeanShift(ali, im2)
        im1 = crop(64)(im1)
        im2 = crop(64)(im2)
        im1 = MeanShift(im1, im2)
        ali = crop(64)(ali)
        ali = MeanShift(ali, im2)
        return im1, im2, ali
    else:
        im1 = crop(256)(im1)
        im2 = crop(256)(im2)
        ali = crop(256)(ali)
        return im1, im2, ali



def computeLoss(im1, im2, ali, tag):
    if tag == 'lr':
        loss_ori = metrics.peak_signal_noise_ratio(im2, im1, data_range=1)
        loss_ali = metrics.peak_signal_noise_ratio(im2, ali, data_range=1)
        is_better = loss_ali > loss_ori
    else:
        loss_ori = lap_match_lose(im1, im2)
        loss_ali = lap_match_lose(ali, im2)
        is_better = loss_ali < loss_ori
    return loss_ori, loss_ali, is_better


def main(x, y):
    total = y - x
    ori_psnr = []
    ali_psnr = []
    ori_loss = []
    ali_loss = []
    lr_cnt = 0
    im_cnt = 0
    plt.figure(20)
    OUT_PATH = r'F:\data\newSR\aligned\moon'
    for i in range(x, y):
        im1 = Image.open(os.path.join(r'F:\data\newSR\origin\moon\lr', f'{i}.tif'))
        im2 = Image.open(os.path.join(r'F:\data\newSR\origin\moon\hr', f'{i}.tif'))
        im1, im2, ali = gen_aligned(im1, im2, (256, 256), 'lr')
        l_o, l_a, good = computeLoss(im1, im2, ali, 'lr')
        print(f'No.{i-x+1}\tori_psnr:{l_o}\tali_psnr:{l_a}\tis_better:{good}')
        ori_psnr.append(l_o)
        ali_psnr.append(l_a)
        lr_cnt += good
        if not good:
            ali = im1
        Image.fromarray(ali).save(os.path.join(OUT_PATH, 'lr', f'{i}.tif'))

        # plt.subplot(total, 3, 3*(i-x)+1)
        # plt.imshow(im1, 'gray')
        # plt.subplot(total, 3, 3*(i-x)+2)
        # plt.imshow(im2, 'gray')
        # plt.subplot(total, 3, 3*(i-x)+3)
        # plt.imshow(ali, 'gray')

        im1 = Image.open(os.path.join(r'F:\data\newSR\origin\moon\img', f'{i}.tif'))
        im2 = Image.open(os.path.join(r'F:\data\newSR\origin\moon\hr', f'{i}.tif'))
        im1, im2, ali = gen_aligned(im1, im2, (1024, 1024), 'im')
        l_o, l_a, good = computeLoss(im1, im2, ali, 'im')
        print(f'No.{i - x + 1}\tori_loss:{l_o}\tali_loss:{l_a}\tis_better:{good}')
        ori_loss.append(l_o)
        ali_loss.append(l_a)
        im_cnt += good

        Image.fromarray(im2).save(os.path.join(OUT_PATH, 'hr', f'{i}.tif'))
        if not good:
            ali = im1
        Image.fromarray(to8(ali)).save(os.path.join(OUT_PATH, 'img', f'{i}.tif'))



    # plt.show()
    ali_psnr = np.array(ali_psnr)
    ali_loss = np.array(ali_loss)
    ori_psnr = np.array(ori_psnr)
    ori_loss = np.array(ori_loss)
    print(f'psnr | origin mean {ori_psnr.mean()} | origin var {ori_psnr.var()} |\npsnr | align mean {ali_psnr.mean()} align var {ali_psnr.var()}')
    print(f'loss | origin mean {ori_loss.mean()} | origin var {ori_loss.var()} |\nloss | align mean {ali_loss.mean()} align var {ali_loss.var()}')
    print(f'num leaped--lr:{lr_cnt / total}, im:{im_cnt / total}')

main(0, 1000)
