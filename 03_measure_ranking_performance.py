from eval import eval_ranking
from ranking import rank_with_euc, rank_with_paired_nn


data_dir = '/Users/ars/data/concatResynData'
data_tag = 'MimMel_cg03real_mfpc19'

rank_devset_path = path.join(data_dir, 'datasets/rankDevel{0}.npz'.format(data_tag))
rank_trainset_path = path.join(data_dir, 'datasets/rankTrain{0}.npz'.format(data_tag))

frames_per_chunk = 11
seed = 12345


avg_rank, ranks, N = eval_rank(rank_with_euc,
                               rank_file=rank_devset_path,
                               n_test=500,
                               frames_per_chunk=frames_per_chunk,
                               seed=seed)
print 'N:', N
print 'avg rank:', avg_rank
print 'ranks:'
print ranks

avg_rank, ranks, N = eval_rank(rank_with_paired_nn,
                               rank_file=rank_devset_path,
                               n_test=500,
                               frames_per_chunk=frames_per_chunk,
                               seed=seed)
print 'N:', N
print 'avg rank:', avg_rank
print 'ranks:'
print ranks

