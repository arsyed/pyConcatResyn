from __future__ import division
import os
from os import path
import numpy as np


def find_files_ext(base_dir, extension):
    found_files = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        if not filenames:
            continue
        matches = [f for f in filenames if f.endswith(extension)]
        if not matches:
            continue
        relpath = dirpath.replace(base_dir, '')
        if relpath.startswith('/'):
            relpath = relpath[1:]
        matches = [path.join(relpath, m) for m in matches]
        found_files.extend(matches)
    return found_files


def mag2db(mag):
    """Convert magnitude to decibels: db = 20 * log10(mag)"""
    # return 20 * np.log10(mag)
    # TODO: check this is okay
    return 10 * np.log10(np.power(mag, 2) + 1e-50)


from scipy.spatial.distance import pdist, squareform
def vector_unique(old, level=0):
    """
    Find the unique rows of matrix old:
        new, newI, oldI = vectorUnique(old)

    Input old is an NxD matrix.  
    Output new is an MxD matrix.  
    Output newI is an N-vector of indices for indexing into new to reproduce old.  
    Output oldI is an M-vector of indices for indexing into old to reproduce new.

    They should satisfy the following conditions:  
        all(sum((new(newI,:) - old).^2, 2) < threshold)
        all(all(new == old(oldI,:)))
    """
    if len(old.shape) == 1:
        old = old[np.newaxis, :]

    threshold = 1e-6 * old.shape[1]

    proj_on = np.random.randn(old.shape[1])
    totals = np.round(1e6 * np.dot(old.astype(np.double), proj_on))
    unq, newI = np.unique(totals, return_inverse=True)
    maxUi = newI.max()
    new = np.zeros((maxUi + 1, old.shape[1]))
    oldI = -1 * np.ones(new.shape[0]).astype(np.int)
    for i in xrange(len(unq)):
        clump, = np.where(newI == i)
        clump_points = old[clump, :]
        if len(clump_points.shape) > 1 and len(clump_points) > 1:
            D = squareform(pdist(clump_points))
        else:
            D = np.array([0])
        if np.mean(D < threshold) > 0.7:
            # these are all the same point
            new[i, :] = old[clump[0], :]
            oldI[i] = clump[0]
        else:
            assert len(unq) > 1, 'Will get caught in infinite recursion.'
            # recurse with another random projection
            newNew, newNewI, newOldI = vector_unique(old[clump, :].astype(np.double), level=level+1)
            newNewI = maxUi + newNewI
            newNewI[newNewI == maxUi] = i
            newOldI = clump[newOldI]
            maxUi = np.max([maxUi, newNewI.max()])

            # incorporate that into results
            newI[clump] = newNewI
            uui = np.unique(newNewI)
            max_uui = uui.max()
            if max_uui >= new.shape[0]:
                new_ext = np.zeros((max_uui - new.shape[0] + 1, new.shape[1]))
                new = np.vstack([new, new_ext])
                oldI_ext = -1 * np.ones(new.shape[0] - len(oldI)).astype(np.int)
                oldI = np.concatenate([oldI, oldI_ext])
            for j in xrange(len(uui)):
                ind, = np.where(uui == uui[j])
                ind = ind[0]
                new[uui[j], :] = newNew[ind, :]
                oldI[uui[j]] = newOldI[ind]

    assert np.all(oldI >= 0)
    oldI = oldI.flatten()
    return new, newI, oldI


def mean_var_normalize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std, mean, std


def mean_var_normalize_test(data, mean, std):
    normed = (data - mean) / std
    normed[~np.isfinite(normed)] = 0.0
    return normed


def sub2ind(shape, idx_rows, idx_cols):
    ind = idx_rowsows * shape[1] + idx_cols
    assert np.all(ind >= 0)
    assert np.all(ind < shape[0] * shape[1])
    return ind









