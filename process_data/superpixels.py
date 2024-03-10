from skimage.segmentation import slic, felzenszwalb
import numpy as np
import cv2.ximgproc as cx
import cv2

# slic
def slicData(data: np.ndarray, n_segments: int = 40, compactness: float = 0.2, sigma: float=0.5):
    """
    使用slic算法分割data
    """
    seg_index = slic(data, n_segments=n_segments, compactness=compactness, sigma=sigma)
    block_num = np.max(seg_index)
    seg_index = seg_index - 1 # 从0开始
    return seg_index, block_num

# felzenszwalb
def felzenszwalbData(data: np.ndarray, scale: float = 120, sigma: float = 0.5, min_size: int = 300):
    """
    使用felzenszwalb算法分割data
    """
    seg_index = felzenszwalb(data, scale=scale, sigma=sigma, min_size=min_size)
    block_num = np.max(seg_index)
    return seg_index, block_num

# lsc
def lscData(data: np.ndarray, region_size: int = 30, ratio: float = 0.075):
    """
    使用lsc算法分割data
    """
    lsc = cx.createSuperpixelLSC(data, region_size=region_size, ratio=ratio)
    lsc.iterate(100)
    lsc_labels = lsc.getLabels()
    lsc_contours = lsc.getLabelContourMask()
    lsc_result = cv2.bitwise_and(data, data, mask=lsc_contours)
    block_num = np.max(lsc_labels)
    return lsc_labels, lsc_contours, lsc_result, block_num

def slicsData(data: np.ndarray, algorithm_name: str, region_size: int = 30, ruler: float = 10.0):
    algorithms = {
        'SLIC':cv2.ximgproc.SLICO,
        'SLICO':cv2.ximgproc.SLIC,
        'MSLIC':cv2.ximgproc.MSLIC,
    }

    slic = cv2.ximgproc.createSuperpixelSLIC(data, algorithm=algorithms[algorithm_name], region_size=region_size, ruler=ruler)
    slic.iterate(100)
    slic_labels = slic.getLabels()
    slic_contours = slic.getLabelContourMask()
    slic_result = cv2.bitwise_and(data, data, mask=slic_contours)
    block_num = np.max(slic_labels)
    return slic_labels, slic_contours, slic_result, block_num

def superpixels(data: np.ndarray, name: str):
    if name == 'SLIC':
        return slicData(data)
    elif name == 'SLICS':
        return slicsData(data, algorithm_name='SLIC')
    elif name == 'SLICO':
        return slicsData(data, algorithm_name='SLICO')
    elif name == 'MSLIC':
        return slicsData(data, algorithm_name='MSLIC')
    elif name == 'Felzenszwalb':
        return felzenszwalbData(data)   
    elif name == 'LSC':
        return lscData(data)