import cc3d
import numpy as np
import matplotlib.pyplot as plt
import  h5py
import cv2
import scipy.signal
import time

start = time.process_time()
#load volum data
Volumen_corrected_path = '/zhome/liuxu/Ground_truth_volume_data/'

volumen = 'R002175-Porsche_Full_04_middle_x_roll.hdf5'

volumen_path1 = Volumen_corrected_path+volumen

f = h5py.File(volumen_path1)

#maximum intensity projection - max slice

start_slice_n = 100

end_slice_n = 200

v = np.array(f['Volume'][start_slice_n:end_slice_n+1, :, :])

v[v<0] = 0

for n in range(v.shape[0]):
    
    cv2.normalize(v[n], v[n], 0, 65535, cv2.NORM_MINMAX)
    
v = (np.rint(v)).astype(np.uint16)
 
v_m = np.amax(v, 0)

#bw-changes

v_m_orig = v_m.copy()

bwchange_deriv = np.zeros(65536)

for y in range(v_m.shape[0]):
        
        for x in range(v_m.shape[1]-1):
            
            if v_m[y, x] != v_m[y, x+1]:
         
                bwchange_deriv[min(v_m[y, x], v_m[y, x+1])] += 1
                
                bwchange_deriv[max(v_m[y, x], v_m[y, x+1])] -= 1

bwchanges = np.cumsum(bwchange_deriv) 

#gray level histogram

hist, bin_edges = np.histogram(v_m, bins=50)

peaks, _ = scipy.signal.find_peaks(hist, width=5, prominence=50)

gray_level =np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])   
        
#find two upper and lower limits

lower_limit = np.argmax(bwchanges)

upper_limit = np.ceil(gray_level[np.max(peaks)]).astype(np.int16)

#find final threshold

thre = np.argmin(bwchanges[lower_limit:upper_limit]) + lower_limit

v_m[v_m_orig>thre]=1
v_m[v_m_orig<=thre]=0
v_m=np.array(v_m,dtype="uint8")

end = time.process_time()
print(end-start)

plt.figure(figsize=(30,30))
plt.title('threshold='+str(thre), size=30)
plt.imshow(v_m)
plt.show()



