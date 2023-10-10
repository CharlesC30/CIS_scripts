import numpy as np
import matplotlib.pyplot as plt
import  h5py
import cv2
import scipy.ndimage

#n = 6
#v_n = np.load('/zhome/liuxu/data_load/R002770-RWTH_Hairpins_Group_03_n01t_slice_261_pin_segmentation/'+ str(n)+'.npy')
#v_n = np.rot90(v_n)
v_n = np.zeros((150, 150))
v_n = cv2.circle(v_n, (75, 75), 50, 100, 3)
#plt.imshow(v_n,'gray')
#plt.show()


'''images to drow profil line'''
v_n = scipy.ndimage.sobel(v_n, 0)
v_n_1 = scipy.ndimage.sobel(v_n, 1)

#profile line
'''line position'''
p_l = v_n[:, 50]
p_l_nor = v_n_1[80, :]
plt.figure(figsize=(60,30))

plt.subplot(1,3,1)
plt.imshow(v_n, "gray")
plt.axvline(x=50,ls="-",c="yellow")

plt.subplot(1,3,2)
plt.imshow(v_n_1, "gray")
plt.axhline(y=80,ls="-",c="yellow")

plt.subplot(1,3,3)
plt.plot(p_l, color='black',linewidth=1.0,linestyle='-',label='vertical')
plt.plot(p_l_nor, color='red',linewidth=1.0,linestyle='-',label='horizontal')
plt.legend(prop={'size':39})
plt.xlabel("coordinate", size=30)
plt.ylabel('gray level value', size=30)
plt.tick_params(labelsize=30)
plt.show()

