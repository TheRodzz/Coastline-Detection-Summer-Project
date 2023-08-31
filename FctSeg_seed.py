"""

Factorization based segmentation with manually selected seeds

"""
import time
import numpy as np
from fseg_filters import image_filtering
import matplotlib.pyplot as plt

from scipy import linalg as LAsci
from skimage import io
from tifffile import imread
from sklearn.cluster import KMeans


def SHcomp(Ig, ws, BinN=11):
    """
    Compute local spectral histogram using integral histograms
    :param Ig: a n-band image
    :param ws: half window size
    :param BinN: number of bins of histograms
    :return: local spectral histogram at each pixel
    """
    h, w, bn = Ig.shape

    # quantize values at each pixel into bin ID
    for i in range(bn):
        b_max = np.max(Ig[:, :, i])
        b_min = np.min(Ig[:, :, i])
        assert b_max != b_min, "Band %d has only one value!" % i

        b_interval = (b_max - b_min) * 1. / BinN
        Ig[:, :, i] = np.floor((Ig[:, :, i] - b_min) / b_interval)

    Ig[Ig >= BinN] = BinN-1
    Ig = np.int32(Ig)

    # convert to one hot encoding
    one_hot_pix = []
    for i in range(bn):
        one_hot_pix_b = np.zeros((h*w, BinN), dtype=np.int32)
        one_hot_pix_b[np.arange(h*w), Ig[:, :, i].flatten()] = 1
        one_hot_pix.append(one_hot_pix_b.reshape((h, w, BinN)))

    # compute integral histogram
    integral_hist = np.concatenate(one_hot_pix, axis=2)

    np.cumsum(integral_hist, axis=1, out=integral_hist)
    np.cumsum(integral_hist, axis=0, out=integral_hist)

    # compute spectral histogram based on integral histogram
    padding_l = np.zeros((h, ws + 1, BinN * bn), dtype=np.int32)
    padding_r = np.tile(integral_hist[:, -1:, :], (1, ws, 1))

    integral_hist_pad_tmp = np.concatenate([padding_l, integral_hist, padding_r], axis=1)

    padding_t = np.zeros((ws + 1, integral_hist_pad_tmp.shape[1], BinN * bn), dtype=np.int32)
    padding_b = np.tile(integral_hist_pad_tmp[-1:, :, :], (ws, 1, 1))

    integral_hist_pad = np.concatenate([padding_t, integral_hist_pad_tmp, padding_b], axis=0)

    integral_hist_1 = integral_hist_pad[ws + 1 + ws:, ws + 1 + ws:, :]
    integral_hist_2 = integral_hist_pad[:-ws - ws - 1, :-ws - ws - 1, :]
    integral_hist_3 = integral_hist_pad[ws + 1 + ws:, :-ws - ws -1, :]
    integral_hist_4 = integral_hist_pad[:-ws - ws - 1, ws + 1 + ws:, :]

    sh_mtx = integral_hist_1 + integral_hist_2 - integral_hist_3 - integral_hist_4

    histsum = np.sum(sh_mtx, axis=-1, keepdims=True) * 1. / bn

    sh_mtx = np.float32(sh_mtx) / np.float32(histsum)

    return sh_mtx


def Fseg(Ig, ws, seeds):
    """
    Factorization based segmentation
    :param Ig: a n-band image
    :param ws: window size for local special histogram
    :param seeds: list of coordinates [row, column] for seeds. each seed represent one type of texture
    :param omega: error threshod for estimating segment number. need to adjust for different filter bank.
    :param nonneg_constraint: whether apply negative matrix factorization
    :return: segmentation label map
    """

    N1, N2, bn = Ig.shape

    ws = int(ws / 2)
    sh_mtx = SHcomp(Ig, ws)

    Z = []
    for seed in seeds:
        Z.append(sh_mtx[seed[0], seed[1], :].reshape((-1, 1)))

    Z = np.hstack(Z)

    Y = sh_mtx.reshape((N1 * N2, -1))

    ZZT = np.dot(Z.T, Z)
    ZZT += np.eye(ZZT.shape[0]) * 1e-6  # Add a small constant to the diagonal
    ZZTinv = LAsci.inv(ZZT)
    Beta = np.dot(np.dot(ZZTinv, Z.T), Y.T)
    ZZTinv= None
    seg_label = np.argmax(Beta, axis=0)
    Beta = None
    return seg_label.reshape((N1, N2))

if __name__ == '__main__':
    i=1
    while(True):
        time0 = time.time()
        # an example of using Fseg
        sar_img = imread('./croppedimages/cropped_image2.tif')
        height, width = sar_img.shape

        sar_img = sar_img.astype(np.float32)
        sar_img = (sar_img - np.min(sar_img)) / (np.max(sar_img) - np.min(sar_img))
        Ig = np.expand_dims(sar_img, axis=2)

        # Compute seeds automatically using K-means
        n_clusters = 3  # number of seeds
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(sar_img.reshape(-1, 1))
        centers = kmeans.cluster_centers_

        # Convert cluster centers to seeds
        seeds = []
        for center in centers:
            coord = np.argmin(np.abs(sar_img - center))
            seeds.append([coord // width, coord % width])

        # run segmentation. try different window size
        seg_out = Fseg(Ig, ws=i, seeds=seeds)

        print ('FSEG runs in %0.2f seconds. ' % (time.time() - time0))


        # Desired resolution
        res = 1  # DPI (dots per inch)

        fig, ax = plt.subplots(figsize=(width, height), dpi=res)

        # show results
        # fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True, figsize=(10, 5))
        # ax[0].imshow(img, cmap='gray')
        seeds = np.array(seeds)
        plt.plot(seeds[:, 1], seeds[:, 0], 'r*')
        ax.imshow(seg_out, cmap='gray')
        seg_out=None
        ax.axis('off')  # Turn off axis labels and ticks for a clean image

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"./croppedimages/results/cropped_image2/ws={i}")
        plt.close()
        print(f"Completed iteration {i}")
        i+=1