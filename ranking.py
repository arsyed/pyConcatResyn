import sys
import numpy as np
from numpy.matlib import repmat
from scipy.spatial.distance import pdist
from blocks.serialization import load
from utils import mean_var_

def rank_with_euc(dicta, queries, gamma=0.1):
    """
    Rank clean dictionary chunks for each noisy query chunk using the paired 
    (clean,noisy) input neural network.
    """
    sim = np.exp(-pdist(queries, metric='wminkowski', p=2, w=dict))
    sim =  sim / (gamma * dict.shape[1])
    best_chunks = np.argmax(sim, axis=0)
    return best_chunks


def rank_with_paired_nn(dicta, queries, model_path, model_opt_path):
    """
    Rank clean dictionary chunks for each noisy query chunk using the paired 
    (clean,noisy) input neural network
    """
    with open(model_path, 'rb') as fmodel:
        ml = load(fmodel)
    model_func = ml.model.get_theano_function()
    npz = np.load(model_opt_path)

    queries = queries.astype(np.float32)
    dicta = dicta.astype(np.float32)

    T = queries.shape[0]
    D = queries.shape[0]
    sim = np.zeros((D, T))
    for t in xrange(T):
        sys.write('.')
        if t % 70 == 0 and t:
            sys.write('\n')
        qrep = repmat(queries[t, :], D, 1)
        text_x = np.hstack([dicta, qrep])
        test_x = mean_var_normalize_test(text_x, npz['train_mean'], npz['train_std'])
        sim[:, t] = model_func(text_x)

    best_chunks = np.argmax(sim, axis=0)
    return best_chunks



