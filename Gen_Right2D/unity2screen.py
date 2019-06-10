import numpy as np
import os  

path = 'unity_train.npy' 
Intrinsic_Matrix = np.array([[1145,0,515,0],[0,1143,512,0],[0,0,1,0]])

Cam = np.load(path)
number_start = 0
number_final = len(Cam)
pixel = np.zeros((number_final,34))
for index in range(number_start,number_final):
	Cam_mid = np.array([Cam[index][0:17] , Cam[index][17:34], Cam[index][34:51], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
	pixel_temp = np.dot(Intrinsic_Matrix, Cam_mid)
	pixel_temp = pixel_temp[:2,:] / pixel_temp[2:3,:]
	pixel[index,0:17] = pixel_temp[0,:]
	pixel[index,17:34] = pixel_temp[1,:]

np.save('unity_train_temp',pixel)
print('finsh')