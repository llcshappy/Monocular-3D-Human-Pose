import os
import numpy as np


def loadData(path):
	
	data = np.array(np.loadtxt(path))
	r, c = data.shape[0], data.shape[1] - 2
	data = data[:,2:53]
	
	return data, r, c

def cam2pixel(fx, fy, cx, cy, frame, camData):

	intrinsic_matrix = [ [fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0] ]	
	
	cam_data = np.zeros((frame,34),dtype=float)
	for index in range(frame):
		cam_mid = [list(camData[index][0:17]), list(camData[index][17:34]), list(camData[index][34:51]), [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] ]
		pixel = np.dot(intrinsic_matrix,cam_mid)
		pixel = pixel[:2,:]/pixel[2:3,:]
		cam_data[index,0:17] = pixel[0,:]
		cam_data[index,17:34] = pixel[1,:]

	return cam_data

def normalizeStats(complete_data):
	
	data_mean = np.mean(complete_data, axis=0)
	data_std  = np.std(complete_data, axis=0)
	
	return data_mean,data_std

def normalizeData(data, data_mean, data_std, frame,joint,dim,data_style):

	data_out = np.zeros((frame,joint), dtype=float)
	#print(data[0])
	for i in range(frame):
		data_out[i,:] = np.true_divide( (data[i] - data_mean), data_std )
		if dim == 2:
			if data_style == 1: #xxyy17
				data_out[i,dim*10] = 0
				data_out[i,dim*10+1] = 0
				data_out[i,dim*10+34] = 0
				data_out[i,dim*10+35] = 0
			if data_style == 2: #xyxy17
				data_out[i,dim*20] = 0
				data_out[i,dim*20+1] = 0
				data_out[i,dim*20+2] = 0
				data_out[i,dim*20+3] = 0
			
		if dim == 3:
			data_out[i,dim*10] = 0
			data_out[i,dim*10+1] = 0
			data_out[i,dim*10+2] = 0
	#print(data_out[0])
	return data_out

def get_all_batches(data_x, data_y, frame, batch_size, training=True ):

	encoder_inputs = np.zeros((frame,68),dtype=float)
	decoder_outputs = np.zeros((frame,51),dtype=float)

	encoder_inputs = data_x
	decoder_outputs = data_y

	if training:
		# Randomly permute everything
		idx = np.random.permutation( frame )
		encoder_inputs  = encoder_inputs[idx, :]
		decoder_outputs = decoder_outputs[idx, :]
		# Make the number of examples a multiple of the batch size
	n_extra  = frame % batch_size
	if n_extra > 0:  # Otherwise examples are already a multiple of batch size
		encoder_inputs  = encoder_inputs[:-n_extra, :]
		decoder_outputs = decoder_outputs[:-n_extra, :]

	n_batches = frame // batch_size
	encoder_inputs  = np.split( encoder_inputs, n_batches )
	decoder_outputs = np.split( decoder_outputs, n_batches )

	return encoder_inputs, decoder_outputs


def unNormalizeData(normalized_data, data_mean, data_std ):

	T = normalized_data.shape[0] # Batch size
	D = data_mean.shape[0] # Dimensionality

	orig_data = np.zeros((T, D), dtype=np.float32)
	orig_data = normalized_data

	# Multiply times stdev and add the mean
	stdMat = data_std.reshape((1, D))
	stdMat = np.repeat(stdMat, T, axis=0)
	meanMat = data_mean.reshape((1, D))
	meanMat = np.repeat(meanMat, T, axis=0)
	orig_data = np.multiply(orig_data, stdMat) + meanMat
	
	return orig_data

def load_h36_data(path):
	
	data = np.array(np.load(path))
	
	return data

def load_pose(isTrain, path):
  
  train_in_frame = 0
  train_in_data = []
  if isTrain:
  	sublist = ['1','5','6','7','8']
  else:
  	sublist = ['9','11']
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
        train_in_data_temp = np.load(path2 + file)
        train_in_frame_temp = train_in_data_temp.shape[0]
        train_in_myjoint = train_in_data_temp.shape[1]
        for i in range(train_in_frame_temp):
        	train_in_data.append(train_in_data_temp[i][0:train_in_myjoint])
        	train_in_frame += 1
  
  return train_in_data, train_in_frame, train_in_myjoint

def root_data(data,dim):
	
	for i in range(len(data)): 
		hip_point_x_d = data[i][dim*10]
		hip_point_y_d = data[i][dim*10+1]
		if dim == 3:
			hip_point_z_d = data[i][dim*10+2]
		for j in range(17):
			data[i][j*dim] -= hip_point_x_d
			data[i][j*dim+1] -= hip_point_y_d
			if dim ==3 :
				data[i][j*3+2] -= hip_point_z_d

	print( "done root data." )

	return data


def xy17_xyr17_to_xy_rxy_17(data,frame):
	
	data_out = []
	for i in range(frame):
		temp = []
		for j in range(17):
			temp.append(data[i][j*2])
			temp.append(data[i][j*2+1])
			temp.append(data[i][j*2+34])
			temp.append(data[i][j*2+35])
		data_out.append(temp)
	return data_out

def xy17_xyr17_to_xrx_yry_17(data,frame):
	
	data_out = []
	for i in range(frame):
		temp = []
		for j in range(17):
			temp.append(data[i][j*2])
			temp.append(data[i][j*2+34])
			temp.append(data[i][j*2+1])
			temp.append(data[i][j*2+35])
		data_out.append(temp)
	
	return data_out

def merge_lr(data_left, data_right, train_in_frame):
	train_in_data = []
	for i in range(train_in_frame):
		train_temp = []
		train_temp.extend(data_left[i])
		train_temp.extend(data_right[i])
		train_in_data.append(train_temp)
	return train_in_data


