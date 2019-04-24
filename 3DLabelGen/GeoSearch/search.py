from numpy import *
import numpy as np
import os

real_depth=0
f_x=1145.049
f_y=1143.781

Cam2 = load_pose(False, path, action)
label1 = load_pose(False,path,action)
number_start=0
number_final=25*int(len(Cam2)/25-1)
lenth= number_final - number_start
print("picture number is %d"%lenth)
final_3D=mat(zeros((number_final-number_start,51)))
index=number_start

for index1 in range(number_start,number_final):	
	index = index1 * 25
	label=mat([label1[index][0:17],label1[index][17:34]])
	label-=label[:,10]
	Cam=mat([Cam2[index][0:17],Cam2[index][17:34],Cam2[index][34:51]])
	Cam_mid=mat(zeros((3,17)))
	modulus_min=2000
	Cam_min = None

	for depth in range(2800,7200):
		Cam_mid[2,:]=Cam[2,:]+depth
		modulus=0
		for i in range(0,17):
			m=label[0,i]/f_x
			n=label[1,i]/f_y
			Cam_mid[0,i]=m*Cam_mid[2,i]
			Cam_mid[1,i]=n*Cam_mid[2,i]
		Cam_mid[0:3,:]-=Cam_mid[0:3,10]

		for i in range(0,17):
			modulus_mid=abs(sqrt((Cam_mid[0,i]-Cam[0,i])**2+(Cam_mid[1,i]-Cam[1,i])**2))
			modulus+=modulus_mid

		if modulus<modulus_min:
			modulus_min=modulus
			real_depth=depth	
			
	for i in range(0,17):
		m=label[0,i]/f_x
		n=label[1,i]/f_y
		Cam[2,i]=Cam[2,i]+real_depth
		Cam[0,i]=m*Cam[2,i]
		Cam[1,i]=n*Cam[2,i]
	Cam-=Cam[:,10]

	Cam[0:3,:]=Cam[0:3,:]-Cam[0:3,10]

			net_loss=abs(Cam_ori - Cam_label)

			net_loss_ave=0
			for i in range(0,17):
				net_loss_ave=net_loss_ave+sqrt(net_loss[0,i]**2+net_loss[1,i]**2+net_loss[2,i]**2)
			net_loss_ave=net_loss_ave/17
			net_loss_final=net_loss_final+net_loss_ave
			ser_loss=abs(Cam-Cam_label)
			
			ser_loss_ave=0
			for i in range(0,17):
				ser_loss_ave=ser_loss_ave+sqrt(ser_loss[0,i]**2+ser_loss[1,i]**2+ser_loss[2,i]**2)
			ser_loss_ave=ser_loss_ave/17
			ser_loss_final=ser_loss_final+ser_loss_ave
			
			o_loss=net_loss_ave- ser_loss_ave
			print "number: %d , net_loss_ave is %f , ser_loss_ave is %f ,loss_net is %f ,loss_ser is %f" %(inx+ number_start, net_loss_ave,ser_loss_ave,net_loss_final/((inx-1)/25+1),ser_loss_final/((inx-1)/25+1) )
			


def load_pose(isTrain, path，action):
  
	train_in_frame = 0
	train_in_data = []
	if isTrain:
 		sublist = ['1','5','6','7','8']
	else:
		sublist = ['9','11']
	for sub in sublist:
		if sub == '.DS_Store':
			continue
		path1 = path + sub + '/' +action + '/'
		filelist = os.listdir(path1)
			for file in filelist:
				train_in_data_temp = np.load(path1 + file)
				train_in_frame_temp = train_in_data_temp.shape[0]
				train_in_myjoint = train_in_data_temp.shape[1]
				for i in range(train_in_frame_temp):
					train_in_data.append(train_in_data_temp[i][0:train_in_myjoint])
  
	return train_in_data



