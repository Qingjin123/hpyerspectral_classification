

__version__ = '0.1.0'
__author__ = 'Qingjin'


from process_data.data_processor import normData, countLabel, sampleMask
from process_data.superpixels import superpixels, showSuperpixels

'''
    superpixels name:
        Felzenszwalb
        SLIC
        SLICS
        SLICO
        MSLIC
        LSC
'''