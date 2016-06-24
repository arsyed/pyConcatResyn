from os import path
from collections import OrderedDict

import theano
from theano import config, function, tensor
config.exception_verbosity = 'high'

import numpy as np
import scipy.io

from blocks.bricks import MLP, Linear, Rectifier, Softmax, Logistic
from blocks.bricks.base import application, Brick
from blocks.bricks.cost import CostMatrix
from blocks.bricks.cost import CategoricalCrossEntropy, BinaryCrossEntropy, MisclassificationRate
from blocks.roles import WEIGHT, OUTPUT, INPUT
from blocks.graph import ComputationGraph, apply_dropout
from blocks.filter import VariableFilter
from blocks.algorithms import GradientDescent, Scale, AdaGrad, CompositeRule, Momentum
from blocks.initialization import IsotropicGaussian, Constant, Uniform, NdarrayInitialization
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks.main_loop import MainLoop
from blocks.model import Model

from fuel.streams import DataStream
from fuel.datasets import IterableDataset, IndexableDataset
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.transformers import Flatten


class MyBinaryCrossEntropy(CostMatrix):
    @application
    def cost_matrix(self, y, y_hat):
        #yh = tensor.nnet.sigmoid(y_hat)
        eps = 1e-12
        yh = tensor.clip(y_hat, eps, 1.0 - eps)
        cost = tensor.nnet.binary_crossentropy(yh, y)
        return cost


class MyIsotropicGaussian(IsotropicGaussian):
    def __init__(self, weight, *args, **kwargs):
        super(MyIsotropicGaussian, self).__init__(*args, **kwargs)
        self.weight = weight
    
    def generate(self, rng, shape):
        gen = super(MyIsotropicGaussian, self).generate(rng, shape)
        gen = gen * self.weight
        # L2 normalization
        gen = gen / np.tile(np.sqrt((gen ** 2).sum(axis=1)).reshape((gen.shape[0], 1)), gen.shape[1])
        return gen


def make_ds(xs, ys, batch_size, n_examples, scheme):
    indexables = OrderedDict([('features', xs), ('targets', ys)])
    sources=('features', 'targets')
    ds = DataStream(IndexableDataset(indexables=indexables, 
                                     sources=sources),
                    iteration_scheme=scheme(batch_size=batch_size, 
                                            examples=n_examples))
    return Flatten(ds)


def train_paired_dnn(train_x, train_y, dev_x, dev_y, test_x, test_y):
    train_y = train_y.flatten().astype(int)
    dev_y = dev_y.flatten().astype(int)
    test_y = test_y.flatten().astype(int)

    batch_size = 256

    n_train, in_dim = train_x.shape
    n_dev = dev_x.shape[0]
    n_test = test_x.shape[0]

    hid_dims = 2 * np.array([512, 512, 512, 512]) 
    out_dim = 1

    ds_train = make_ds(train_x, train_y, batch_size, n_train, SequentialScheme)
    ds_dev = make_ds(dev_x, dev_y, batch_size, n_dev, SequentialScheme)
    ds_test = make_ds(test_x, test_y, batch_size, n_test, SequentialScheme)

    mlp = MLP(
      activations=[Rectifier(), Rectifier(), Rectifier(), Rectifier(), Logistic()],
      dims=[in_dim, hid_dims[0], hid_dims[1], hid_dims[2], hid_dims[3], out_dim],
      weights_init=Uniform(mean=0, width=1/32),
      biases_init=Constant(0)
    )
    mlp.initialize()

    x = tensor.matrix('features')
    y = tensor.matrix('targets', dtype='int64')
    y_hat = mlp.apply(x)
    model = Model(y_hat)

    cost = MyBinaryCrossEntropy().apply(y, y_hat)
    cost.name = 'cost'
    misrate = MisclassificationRate().apply(y.flatten(), y_hat)
    misrate.name = 'misclassfication'

    cg = ComputationGraph([cost, misrate])
    drop_vars = VariableFilter(
        roles=[INPUT], 
        bricks=mlp.linear_transformations[1:]
    )(cg.variables)
    cg_dropout = apply_dropout(cg, drop_vars, 0.2)
    cost_dropout, error_rate_dropout = cg_dropout.outputs

    learning_rate = 0.0015
    momentum = 0.9
    step_rule = CompositeRule([
        Momentum(learning_rate=learning_rate, momentum=momentum),
        AdaGrad(learning_rate=learning_rate)
    ])
    algorithm = GradientDescent(cost=cost_dropout, 
                                parameters=cg.parameters,
                                step_rule=step_rule)

    monitor_train = TrainingDataMonitoring(
        variables=[cost_dropout, 
                   error_rate_dropout, 
                   aggregation.mean(algorithm.total_gradient_norm)],
        after_epoch=True,
        prefix="train"
    )
    monitor_dev = DataStreamMonitoring(
    #    variables=[cost_dropout, error_rate_dropout],
        variables=[cost, misrate],
        data_stream=ds_dev, 
        prefix="dev"
    )
    monitor_test = DataStreamMonitoring(
    #    variables=[cost_dropout, error_rate_dropout],
        variables=[cost, misrate],
        data_stream=ds_test, 
        prefix="test"
    )
    track_str = 'train_{0}'.format(cost_dropout.name)
    track_best_str = '{0}_best_so_far'.format(track_str)
    print track_str, track_best_str

    n_epochs = 2
    print 'n_epochs:', n_epochs
    main_loop = MainLoop(
       model=model,
       data_stream=ds_train, 
       algorithm=algorithm,
       extensions=[Timing(),
                   monitor_train,
                   monitor_dev,
                   monitor_test,
                   TrackTheBest(track_str),
                   Checkpoint("best_model.pkl",
                       use_cpickle = True
                   ).add_condition(['after_epoch'],
                       predicate=OnLogRecord(track_best_str)),
                   FinishAfter(after_n_epochs=n_epochs), 
                   # FinishIfNoImprovementAfter(track_best_str, epochs=n_epochs),
                   Printing()]
    )
    main_loop.run() 

    acc([x], y_hat, train_x, train_y, 'train')
    acc([x], y_hat, dev_x, dev_y, 'dev')
    acc([x], y_hat, test_x, test_y, 'test')


def acc(inputs, outputs, features, targets, name):
    fpred = theano.function(inputs=inputs, outputs=outputs)
    preds = fpred(features)
    accuracy = (targets == preds.flatten().round()).sum() / len(targets)
    print 'accuracy on {0}: {1}'.format(name, accuracy)



