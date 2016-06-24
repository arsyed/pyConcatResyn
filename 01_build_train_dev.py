from os import path
from dataprep import build_classification_dataset, convert_classification_to_rank
from featex import load_mel_chunks_from_wav


data_dir = '/Users/ars/data/concatResynData'
data_tag = 'MimMel_cg03real_mfpc19'

dev_clean_wav_dir = path.join(data_dir, 'inputs/reverberated_cg03/devel')
dev_noisy_wav_dir = path.join(data_dir, 'inputs/isolated_cg03/devel')

train_clean_wav_dir = path.join(data_dir, 'inputs/reverberated_cg03/train')
train_noisy_wav_dir = path.join(data_dir, 'inputs/isolated_cg03/train')

devset_path = path.join(data_dir, 'datasets/clsDevel{0}.npz'.format(data_tag))
trainset_path = path.join(data_dir, 'datasets/clsTrain{0}.npz'.format(data_tag))

rank_devset_path = path.join(data_dir, 'datasets/rankDevel{0}.npz'.format(data_tag))
rank_trainset_path = path.join(data_dir, 'datasets/rankTrain{0}.npz'.format(data_tag))

n_files = 2000
n_pos = 100000
frames_per_chunk = 11
seed = 12345


build_classification_dataset(devset_path, dev_clean_wav_dir, dev_noisy_wav_dir,
                             frames_per_chunk=frames_per_chunk,
                             n_files=n_files,
                             n_pos=n_pos,
                             load_chunk_path=load_mel_chunks_from_wav,
                             seed=12345)
print 'built classification dev set:', devset_path

build_classification_dataset(trainset_path, train_clean_wav_dir, train_noisy_wav_dir,
                             frames_per_chunk=frames_per_chunk,
                             n_files=n_files,
                             n_pos=n_pos,
                             load_chunk_path=load_mel_chunks_from_wav,
                             seed=12345)
print 'built classification train set:', trainset_path


convert_classification_to_rank(devset_path, rank_devset_path, frames_per_chunk)
print 'built rank dev set:', rank_devset_path

convert_classification_to_rank(trainset_path, rank_trainset_path, frames_per_chunk)
print 'built rank train set:', rank_trainset_path


