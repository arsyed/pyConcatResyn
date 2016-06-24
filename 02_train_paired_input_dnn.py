# todo: save outputs in model dir 
from os import path
from datetime import datetime
import numpy as np
from dnn_paired import train_paired_dnn
from dataprep import shrink_chunks, split_train_dev
from utils import mean_var_normalize, mean_var_normalize_test


data_dir = '/Users/ars/data/concatResynData'
data_tag = 'MimMel_cg03real_mfpc19'

testset_path = path.join(data_dir, 'datasets/clsDevel{0}.npz'.format(data_tag))
trainset_path = path.join(data_dir, 'datasets/clsTrain{0}.npz'.format(data_tag))

frames_per_chunk = 11
n_parallel_cols = 2

trainset = np.load(trainset_path)
train_x = shrink_chunks(trainset['newCmbChunk16k'].astype(np.float32),
						trainset['frames_per_chunk'], 
						frames_per_chunk,
						n_parallel_cols - 1)
train_y = trainset['newTarget16k'].astype(np.float32)
train_x, train_mean, train_std = mean_var_normalize(train_x)
train_x, train_y, dev_x, dev_y = split_train_dev(train_x, train_y, 0.925)


testset = np.load(testset_path)
test_x = shrink_chunks(testset['newCmbChunk16k'].astype(np.float32),
						testset['frames_per_chunk'], 
						frames_per_chunk,
						n_parallel_cols - 1)
test_y = testset['newTarget16k'].astype(np.float32)
test_x = mean_var_normalize_test(test_x, train_mean, train_std)

model_tag = 'paired_data{0}_fpc{1}'.format(data_tag, frames_per_chunk)
print 'training paired dnn:', datetime.now()
train_paired_dnn(train_x, train_y, dev_x, dev_y, test_x, test_y)
print '... done training paired dnn:', datetime.now()

np.savez_compressed('model_opt.npz', 
					train_mean=train_mean, 
					train_std=train_std)








