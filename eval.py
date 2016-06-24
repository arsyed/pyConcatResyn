import numpy as np
from dataprep import shrink_chunks
from utils import sub2ind


def eval_ranking(rank_func, rank_file, n_test, frames_per_chunk, n_test_per_time=1e4, seed=12345):
    """
    Evaluate the ranking performance of a ranking function on a dataset.

    rank_file should contain three variables: clean, noisy,
    cleanForEachNoisy.  
        clean and noisy are matrices of features.
        cleanForEachNoisy is a vector containing the index of the correct clean
        row for each noisy row.  
    rankFn should have the following interface:
        [bestChunks sim] = rankFn(clean, noisy);
    Outputs avgRank, the average rank of the correct chunk, and ranks, the
    rank of each correct chunk.  Both are between 0 and 1, with 1 being best.
    """
    npz = np.load(rank_file)
    clean = shrink_chunks(npz['clean'], npz['frames_per_chunk'], frames_per_chunk, 0)
    noisy = shrink_chunks(npz['noisy'], npz['frames_per_chunk'], frames_per_chunk, 0)

    perm = np.random.permutation(noisy.shape[1])
    n_noisy = np.min([n_test, noisy.shape[0]])
    noisy = noisy[perm[:n_noisy], :]
    clean_for_each_noisy = npz['clean_for_each_noisy'][:n_noisy, :]

    N = clean.shape[0]
    sim = np.zeros((N, noisy.shape[0]))
    for aat in xrange(0, noisy.shape[0], n_test_per_time):
        ind_att_end = np.min([aat + n_test_per_time, noisy.shape[0] - 1])
        ind_aat = np.arange(aat, ind_att_end)
        _, sim[:, ind_aat] = rank_func(clean, noidy[ind_aat, :])

    sim_of_right = sim[sub2ind(sim.shape, clean_for_each_noisy.T, np.arange(len(clean_for_each_noisy)))]
    ranks = np.mean(sim <= sim_of_right, axis=0)
    avg_ranks = ranks.mean()

    return avg_ranks, ranks, N

