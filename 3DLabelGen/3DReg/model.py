
"""Predicting 3d poses from 2d joints"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py
import copy

import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import linear_model
import data_pre

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 1, "Dropout keep probability. 1 means no dropout")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")
tf.app.flags.DEFINE_integer("epochs", 200, "How many epochs we should train for")
tf.app.flags.DEFINE_boolean("camera_frame", False, "Convert 3d poses to camera coordinates")
tf.app.flags.DEFINE_boolean("max_norm", False, "Apply maxnorm constraint to the weights")
tf.app.flags.DEFINE_boolean("batch_norm", False, "Use batch_normalization")

# Architecture
tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("residual", False, "Whether to add a residual connection every 2 layers")

# Evaluation
tf.app.flags.DEFINE_boolean("evaluateActionWise",False, "The dataset to use either h36m or heva")

# Directories
tf.app.flags.DEFINE_string("cameras_path","data/h36m/cameras.h5","Directory to load camera parameters")
tf.app.flags.DEFINE_string("data_dir",   "data/h36m/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "experiments", "Training directory.")

# Train or load
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("predict_14", False, "predict 14 joints")
tf.app.flags.DEFINE_boolean("use_sh", False, "Use 2d pose predictions from StackedHourglass")

FLAGS = tf.app.flags.FLAGS

train_dir = os.path.join( FLAGS.train_dir,
  'dropout_{0}'.format(FLAGS.dropout),
  'epochs_{0}'.format(FLAGS.epochs) if FLAGS.epochs > 0 else '',
  'lr_{0}'.format(FLAGS.learning_rate),
  'residual' if FLAGS.residual else 'not_residual',
  'depth_{0}'.format(FLAGS.num_layers),
  'linear_size{0}'.format(FLAGS.linear_size),
  'batch_size_{0}'.format(FLAGS.batch_size),
  'maxnorm' if FLAGS.max_norm else 'no_maxnorm',
  'batch_normalization' if FLAGS.batch_norm else 'no_batch_normalization',
  'predict_14' if FLAGS.predict_14 else 'predict_17')

print( train_dir )
summaries_dir = os.path.join( train_dir, "log" ) # Directory for TB summaries

os.system('mkdir -p {}'.format(summaries_dir))

def save_3d_file(sub,action,fileName,dataSet):
  data_dir = '/Users/chenyan/Documents/sensetime/experiment_unity_RIMG/2d-pose-right-baseline/data/2D_right/' 
  #print(dataSet)
  filePath = data_dir+sub+'/'+action+'/'
  folder = os.path.exists(filePath)
  if not folder:
    os.makedirs(filePath)
  np.save(filePath+fileName,dataSet)


def create_model( session, batch_size ):
  """
  Create model and initialize it or load its parameters in a session

  Args
    session: tensorflow session
    actions: list of string. Actions to train/test on
    batch_size: integer. Number of examples in each batch
  Returns
    model: The created (or loaded) model
  Raises
    ValueError if asked to load a model, but the checkpoint specified by
    FLAGS.load cannot be found.
  """

  model = linear_model.LinearModel(
      FLAGS.linear_size,
      FLAGS.num_layers,
      FLAGS.residual,
      FLAGS.batch_norm,
      FLAGS.max_norm,
      batch_size,
      FLAGS.learning_rate,
      summaries_dir,
      FLAGS.predict_14,
      dtype=tf.float16 if FLAGS.use_fp16 else tf.float32)
   

  if FLAGS.load <= 0:
    # Create a new model from scratch
    print("Creating model with fresh parameters.")
    session.run( tf.global_variables_initializer() )
    return model

  # Load a previously saved model
  ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")
  print( "train_dir", train_dir )

  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific checkpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.join( os.path.join(train_dir,"checkpoint-{0}".format(FLAGS.load)) )
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    model.saver.restore( session, ckpt.model_checkpoint_path )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model

def predict():

  # Avoid using the GPU if requested
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto(
    device_count=device_count,
    allow_soft_placement=True )) as sess:

    # === Create the model ===
    print("Creating %d bi-layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    model = create_model( sess, FLAGS.batch_size )
    model.train_writer.add_graph( sess.graph )
    print("Model created")

    path = '/Users/chenyan/Documents/sensetime/experiment_unity_RIMG/2d-pose-right-baseline/data/save3D_new/' 
    sublist = os.listdir(path) 
    
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
          test_in_data = np.load(path2 + file)
          test_in_frame = test_in_data.shape[0]
          test_in_cam_data = data_pre.cam2pixel(1145, 1143, 515, 512, test_in_frame, test_in_data)
          test_in_data_mean, test_in_data_std = data_pre.normalizeStats(test_in_cam_data)
          test_in_data_norm = data_pre.normalizeData(test_in_cam_data, test_in_data_mean, test_in_data_std, test_in_frame)
          print( "done reading and normalizing data of %s."%file )

          dp = 1.0 # dropout keep probability is always 1 at test time
          enc_in = np.zeros((test_in_frame, 34), dtype=np.float32)
          enc_in = test_in_data_norm
          _, _, poses2d = model.step( sess, enc_in, enc_in, dp, isTraining=False )
          # denormalize
          enc_in  = data_pre.unNormalizeData( enc_in, test_in_data_mean, test_in_data_std )
          poses2d = data_pre.unNormalizeData( poses2d, test_in_data_mean, test_in_data_std )
          save_3d_file(sub,action,file,poses2d)
          print('finsh %s, %s, %s'%(sub,action,file))

  

def main(_):
  predict()


if __name__ == "__main__":
  tf.app.run()
  


