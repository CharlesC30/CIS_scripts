import cc3d
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage
import h5py
import cv2
import os


'''for 8 bit image'''
def cut_pin_2d(img):

    v_n = img

    v_n_orig = v_n.copy()

    label_n = []

    thre = range(0, 256)

    for t in thre:
        
        v_n[v_n_orig > t] = 1
        
        v_n[v_n_orig <= t] = 0
        
        v_n = np.array(v_n,dtype="bool")
        
        labels_out, N = cc3d.connected_components(v_n, return_N=True)
        
        label_n.append(N)

    thre = np.array(thre)

    label_n = np.array(label_n)
    
    # plt.plot(thre, label_n)
    # plt.show()
    
    hist, bin_edges = np.histogram(v_n_orig, bins=100)

    peaks, _ = scipy.signal.find_peaks(hist, width=3, prominence=50)
    
    # plt.plot(hist)
    # plt.show()

    gray_level =np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
    
    lower_limit = np.argmax(label_n)
    
    upper_limit = np.ceil(gray_level[np.max(peaks)]).astype(np.int64)
    
    t = np.argmin(label_n[lower_limit:upper_limit]) + lower_limit

    print(t)

    v_n[v_n_orig > t] = 1

    v_n[v_n_orig <= t] = 0

    v_n = np.array(v_n, dtype="bool")

    np.save("slice30_thresh", v_n)

    return v_n




# Volumen_corrected_path = '/zhome/liuxu/Ground_truth_volume_data/'

# volumen = 'R002323-Porsche_Stator_03_full_50W_x_roll.hdf5'

# volumen_path1 = Volumen_corrected_path+volumen

# f = h5py.File(volumen_path1)

# slice_n = 128

# v_n = np.array(f['Volume'][slice_n, :, :])
os.chdir("/lhome/clarkcs/Cu-pins")
v_n = np.load("slice30.npy")
v_n[v_n<0] = 0

cv2.normalize(v_n, v_n, 0, 256, cv2.NORM_MINMAX)


output = cut_pin_2d(v_n)
plt.imshow(output)
plt.show()
