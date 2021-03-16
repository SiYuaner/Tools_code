import cv2
import numpy as np
import matplotlib.pyplot as plt


def tensor_to_np(tensor):
    # img = tensor.mul(255).byte()
    img = tensor.byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

def show_from_cv(img, title=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    # plt.figure()
    # plt.imshow(img)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.001)

def img_tensor2img(tensor):
    img = tensor_to_np(tensor)
    img = show_from_cv(img)
    return img

def label_tensor2img(tensor):
    img = tensor_to_np(tensor) * 255
    # img = show_from_cv(img)
    return img