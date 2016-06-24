from __future__ import division
import os, sys
from os import path
import numpy as np
from featex import load_mel_chunks_from_wav
import utils


def build_classification_dataset(out_file,
                                 clean_wav_dir,
                                 noisy_wav_dir,
                                 frames_per_chunk,
                                 n_files=275,
                                 n_pos=100000,
                                 load_chunk_fn=load_mel_chunks_from_wav,
                                 seed=12345):
    """
    Params
    ------

    n_files:    number of wav files to choose for building dataset
    n_pos:      number of positive examples to select. final matrix 
                will have 2*n_pos examples (positive and negative).
    """
    np.random.seed(seed)

    # find .wav files in subdirectories of clean and noisy dirs
    cf = utils.find_files_ext(clean_wav_dir, '.wav')
    nf = utils.find_files_ext(noisy_wav_dir, '.wav')
    # only keep files that are common to clean and noisy sets
    files = np.array(list(set(cf) & set(nf)))

    # if we have more files than requested, randomly choose n_files
    if len(files) > n_files:
        np.random.choice(files, n_files)
        print 'kept {0}/{1} files'.format(n_files, len(files))
    else:
        print 'kept all {0} files'.format(len(files))

    cc, nc = [], []
    file_ind, row_ind = [], []
    for i, f in enumerate(files):
        sys.stdout.write('.')
        if i % 70 == 0:
            sys.stdout.write('\n')

        cfi = path.join(clean_wav_dir, f)
        nfi = path.join(noisy_wav_dir, f)

        # chunks wavChunks x fs hop win
        clean_chunks, clean_wav_chunks, xc, fs, hop, win = load_chunk_fn(cfi, frames_per_chunk)
        noisy_chunks, noisy_wav_chunks, xn, fs, hop, win = load_chunk_fn(nfi, frames_per_chunk)
        assert xc.shape == xn.shape
        assert clean_chunks.shape == noisy_chunks.shape

        assert np.all(np.isfinite(clean_chunks))
        assert np.all(np.isfinite(noisy_chunks))
        cc.append(clean_chunks)
        nc.append(noisy_chunks)

        # TODO: check if this works out: file_ind and row_ind are zero based unlike matlab
        fi = i * np.ones(clean_chunks.shape[0])
        file_ind.append(fi)        
        # transposed array; m x 1
        #ri = np.arange(1, clean_chunks.shape[0]+ 1)[:, np.newaxis]
        ri = np.arange(clean_chunks.shape[0])
        row_ind.append(ri)
    sys.stdout.write('\n')

    cc = np.vstack(cc).astype(np.float32)
    nc = np.vstack(nc).astype(np.float32)
    assert np.all(np.isfinite(cc))
    assert np.all(np.isfinite(nc))
    file_ind = np.concatenate(file_ind)
    row_ind = np.concatenate(row_ind)
    N = cc.shape[0]

    # subsample if necessary
    if N > n_pos:
        indices = np.arange(N)
        np.random.shuffle(indices)
        keep = np.sort(indices[:n_pos])
        cc = cc[keep, :]
        nc = nc[keep, :]
        assert np.all(np.isfinite(cc))
        assert np.all(np.isfinite(nc))
        file_ind = file_ind[keep]
        row_ind = row_ind[keep]
        N = cc.shape[0]
        print 'kept {0}/{1} data points'.format(N*2), len(indices)*2
    else:
        print 'kept all {0} data points'.format(N*2)

    # shuffle clean rows to make negative examples
    ri = np.random.randint(0, N, N)
    for idx in xrange(len(nc)):
        while idx == ri[idx]:
            ri[idx] = np.random.randint(0, N)
    ncc = cc[ri, :]
    assert np.all(np.isfinite(ncc))

    # assemble final matrices
    newCmbChunk16k = np.vstack([ np.hstack([cc, nc]), np.hstack([ncc, nc]) ])
    newTarget16k = np.concatenate([np.ones(N), np.zeros(N)])
    assert np.all(np.isfinite(newCmbChunk16k))
    assert np.all(np.isfinite(newTarget16k))
    cleanFileInd = np.concatenate([file_ind, file_ind[ri]])
    cleanRowInd = np.concatenate([row_ind, row_ind[ri]])
    noisyFileInd = np.concatenate([file_ind, file_ind])
    noisyRowInd = np.concatenate([row_ind, row_ind])

    np.savez_compressed(out_file,
                        newCmbChunk16k=newCmbChunk16k,
                        newTarget16k=newTarget16k,
                        fs=fs,
                        hop=hop,
                        win=win,
                        files=files,
                        clean_wav_dir=clean_wav_dir,
                        noisy_wav_dir=noisy_wav_dir,
                        seed=seed,
                        file_ind=file_ind,
                        row_ind=row_ind,
                        cleanFileInd=cleanFileInd,
                        cleanRowInd=cleanRowInd,
                        noisyFileInd=noisyFileInd,
                        noisyRowInd=noisyRowInd,
                        frames_per_chunk=frames_per_chunk)


from utils import vector_unique
def convert_classification_to_rank(in_file, out_file, frames_per_chunk):
    """
    Convert classification supervision data to ranking.

    Ranking supervision is: 
        a matrix of unique clean chunks, 
        a matrix of unique noisy chunks, 
        and a vector of the single correct clean chunk for each noisy chunk.  
    Saves three variables, noisy, clean, and cleanForEachNoisy.  
    The matrix clean(cleanForEachNoisy,:) has rows corresponding to noisy.
    """
    npz = np.load(in_file)
    newCmbChunk16k = npz['newCmbChunk16k']
    newTarget16k = npz['newTarget16k']

    keep = newTarget16k > 0
    feat = newCmbChunk16k[keep, :]

    mid = feat.shape[1] // 2
    old_clean = feat[:, :mid]
    noisy = feat[:, mid:]

    clean, clean_for_each_noisy, _ = vector_unique(old_clean)
    assert np.all(np.abs(clean[clean_for_each_noisy, :] - old_clean) <= 1e-5)

    np.savez_compressed(out_file, 
                        noisy=noisy, 
                        clean=clean, 
                        clean_for_each_noisy=clean_for_each_noisy,
                        frames_per_chunk=frames_per_chunk)


def split_train_dev(train_x, train_y, pct):
    perm = np.random.permutation(train_x.shape[0])
    train_x = train_x[perm, :]
    train_y = train_y[perm]
    cut = int(np.round(pct * train_x.shape[0]))
    dev_x = train_x[(cut + 1):, :]
    dev_y = train_y[(cut + 1):]
    train_x = train_x[:(cut + 1), :]
    train_y = train_y[:(cut + 1)]
    return train_x, train_y, dev_x, dev_y


def shrink_one_chunk(old, old_fpc, new_fpc):
    new_fpc = np.min([new_fpc, old_fpc])
    nF = old.shape[1] / old_fpc

    first_maj_col = np.floor((old_fpc - new_fpc) / 2)
    last_maj_col = new_fpc + first_maj_col
    keep_cols = np.arange(first_maj_col * nF, last_maj_col * nF).astype(np.int)

    new = old[:, keep_cols]
    return new


def shrink_chunks(old, old_fpc, new_fpc, extra_parallel_cols=0):
    """
    Take a subset of frames from matrix of unrolled chunks.

    Params
    ------

    old:        matrix of old chunks
    old_fpc:    frames per chunk in old
    new_fpc:    new frames per chunk
    """
    C = old.shape[1]
    n_parallel_cols = extra_parallel_cols + 1
    assert C % n_parallel_cols == 0, (
        'Number of columns {0} not divisible by {1}'.format(C, n_parallel_cols))

    oneC = C / n_parallel_cols
    boundaries = np.arange(0, C, oneC)
    new = []
    for c in xrange(len(boundaries)):
        inds = np.arange(boundaries[c], boundaries[c] + oneC).astype(np.int)
        new.append(shrink_one_chunk(old[:, inds], old_fpc, new_fpc))
    new = np.hstack(new)
    return new












