import numpy as np
import os  

path = '/home/sensetime/Documents/experiment_unity_RIMG/2d-pose-right-baseline-h36m/data/3d_right/' 
sublist = os.listdir(path) 
Intrinsic_Matrix = np.array([[1145,0,515,0],[0,1143,512,0],[0,0,1,0]])

def save_3d_file(sub,action,fileName,dataSet):
	data_dir = '/home/sensetime/Documents/experiment_unity_RIMG/2d-pose-right-baseline-h36m/data/2d_right/' 
 	#print(dataSet)
	filePath = data_dir+sub+'/'+action+'/'
	folder = os.path.exists(filePath)
	if not folder:
		os.makedirs(filePath)
	np.save(filePath+fileName,dataSet)

for sub in sublist:
	if sub == '.DS_Store':
		continue
	path1 = path + sub + '/'
	actionList= os.listdir(path1)
	for action in actionList:
		if action == '.DS_Store':
			continue
		path2 = path1 + action + '/'
		filelist = os.listdir(path2)
		for file in filelist:
			Cam=np.load(path2 + file)
			number_start=0
			number_final=len(Cam)
			pixel = np.zeros((number_final,34))
			for index in range(number_start,number_final):
				Cam_mid=np.array([Cam[index][0:17],Cam[index][17:34],Cam[index][34:51],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
				pixel_temp = np.dot(Intrinsic_Matrix,Cam_mid)
				pixel_temp = pixel_temp[:2,:] / pixel_temp[2:3,:]
				pixel[index,0:17] = pixel_temp[0,:]
				pixel[index,17:34] = pixel_temp[1,:]
			save_3d_file(sub,action,file,pixel)
		
print('finsh')