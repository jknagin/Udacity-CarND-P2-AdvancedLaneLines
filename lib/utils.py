import cv2
import matplotlib.pyplot as plt
from typing import Tuple

CMAP = 'gray'


def read_image(fname: str):
    return cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)


def save_image(fname: str, img, binary: bool = False):
    if binary:
        plt.imsave(fname, img, cmap="gray")
    else:
        cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def hls(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def plot_two_images(img1, img2, title1: str = "First Image", title2: str = "Second Image",
                    binary: Tuple[bool, bool] = (False, False)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    fig.tight_layout()
    ax1.imshow(img1, cmap=CMAP, vmin=0, vmax=1) if binary[0] else ax1.imshow(img1)
    ax2.imshow(img2, cmap=CMAP, vmin=0, vmax=1) if binary[1] else ax2.imshow(img2)
    ax1.set_title(title1, fontsize=30)
    ax2.set_title(title2, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def show():
    plt.show()
