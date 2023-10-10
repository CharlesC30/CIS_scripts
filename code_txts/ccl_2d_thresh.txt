import cc3d
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage

# modified version of Xingyu's CCL-2D.py to return threshold and
# not alter input image

def ccl_threshold(img):

    v_n = img.copy()

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

    return t