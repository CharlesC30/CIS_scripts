import cc3d
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage
import h5py
import cv2


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
    
    plt.plot(thre, label_n)
    plt.show()
    
    hist, bin_edges = np.histogram(v_n_orig, bins=1000)

    peaks, _ = scipy.signal.find_peaks(hist, width=3, prominence=50)
    
    plt.plot(hist)
    plt.show()

    gray_level =np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
    
    lower_limit = np.argmax(label_n)
    
    upper_limit = np.ceil(gray_level[np.max(peaks)]).astype(np.int64)

    print(lower_limit, upper_limit)
    
    t = np.argmin(label_n[lower_limit:upper_limit]) + lower_limit

    v_n[v_n_orig > t] = 1

    v_n[v_n_orig <= t] = 0

    v_n = np.array(v_n, dtype="bool")

    return v_n




Volumen_corrected_path = '/lhome/clarkcs/Cu-pins/from_Xingyu/Samples/'

volumen = 'R002770-RWTH_Hairpins_Group_03_n01t.hdf5'

volumen_path1 = Volumen_corrected_path+volumen

f = h5py.File(volumen_path1)

slice_n = 260

v_n = np.array(f['Volume'][slice_n, :, :])
# v_n = np.load("/lhome/clarkcs/Cu-pins/pin-pore-examples/ict_pin_pore.npy")
plt.imshow(v_n)
plt.show()
v_n[v_n<0] = 0

cv2.normalize(v_n, v_n, 0, 256, cv2.NORM_MINMAX)


output = cut_pin_2d(v_n)
plt.imshow(output)
plt.show()

