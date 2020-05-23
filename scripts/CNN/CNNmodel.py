import argparse

import numpy as np
import chainer
from chainer import report, training, Chain, datasets, iterators, optimizers, cuda, Reporter, report_scope
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import matplotlib.pyplot as plt
plt.switch_backend('agg')


from chainer import serializers



class TestModeEvaluator(extensions.Evaluator):
    
    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret

class CNN(Chain):
    def __init__(self, output_ch1, output_ch2, filter_height, filter_width, n_units, n_label):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(1, output_ch1, (filter_height, filter_width)),
            bn1 = L.BatchNormalization(output_ch1),
            conv2 = L.Convolution2D(output_ch1, output_ch2, (filter_height, 1)),
            bn2 = L.BatchNormalization(output_ch2),
            fc1 = L.Linear(None, n_units),
            fc2 = L.Linear(None, n_label))

    def __call__(self, x, train=True):
        h1 = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), ksize=(10,1), stride=8)
        h2 = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h1))), ksize=(14,1), stride=8)
        h3 = F.dropout(F.relu(self.fc1(h2)), ratio=0.5)
        y = self.fc2(h3)
        return y

    def extract(self, x):
        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                h1 = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), ksize=(10, 1), stride=8)
                h2 = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h1))), ksize=(14, 1), stride=8)
                h3 = F.dropout(F.relu(self.fc1(h2)), ratio=0.5)

        return h3

class SClassifier(Chain):
    def __init__(self, predictor, lastlayer):
        super(SClassifier, self).__init__(predictor=predictor)
        self.lastlayer = lastlayer
    def __call__(self, x, t, train=False):
        y = self.predictor(x)

        if self.lastlayer == 1:     # The number of last layer units = 1
            loss = F.sigmoid_cross_entropy(y, t.reshape(len(t),1))
            accuracy = F.binary_accuracy(y, t.reshape(len(t),1))
            f1 = F.f1_score(y, t)
        else:                       # The number of last layer units = 2
            loss = F.softmax_cross_entropy(y, t)
            accuracy = F.accuracy(y, t).data
            f1 = F.f1_score(y, t)[0].data
            # f1 = F.f1_score(y, t)
        summary = F.classification_summary(y, t, beta = 1.0)
        precision = summary[0]
        recall = summary[1]
        f_value = summary[2]

        return accuracy.min(), f1[0]


class MyClassifier(Chain):
    def __init__(self, predictor, lastlayer):
        super(MyClassifier, self).__init__(predictor=predictor)
        self.lastlayer = lastlayer

    def __call__(self, x, t):
        y = self.predictor(x)
        if self.lastlayer == 1:     # The number of last layer units = 1
            loss = F.sigmoid_cross_entropy(y, t.reshape(len(t), 1))
            accuracy = F.binary_accuracy(y, t.reshape(len(t), 1))
        else:                       # The number of last layer units = 2
            loss = F.softmax_cross_entropy(y, t)
            accuracy = F.accuracy(y, t)
        summary = F.classification_summary(y, t, beta = 1.0)
        precision = summary[0]
        recall = summary[1]
        f_value = summary[2]
        reporter = Reporter()
        observer = object()
        reporter.add_observer('f_value:', observer)
        observation={}    
        with reporter.scope(observation):
            reporter.report({'x': f_value}, observer)
        report({'loss': loss,
                'accuracy': accuracy, 
                'precision': precision,
                'recall': recall,
                'f_value': f_value}, self)
        report(dict(('f_value_%d' % i, val) for i, val in enumerate(f_value)), self)

        return loss


class Runchainer:
    def __init__(self, batchsize, outdir, output_ch1, output_ch2, filter_height, width, n_units, n_label):
        self.batchsize = batchsize
        self.outdir = outdir
        self.output_ch1 = output_ch1
        self.output_ch2 = output_ch2
        self.filter_height = filter_height
        self.width = width
        self.n_units = n_units
        self.n_label = n_label

    def learn(self, train, test, gpu, epoch, cnn_model_path):

        self.epoch = epoch

        ## Set up model =============================================
        model = MyClassifier(CNN(self.output_ch1, self.output_ch2, self.filter_height, \
                                 self.width, self.n_units, self.n_label), self.n_label)
        # Set up GPU ------------------------------------------
        if gpu >= 0:
            chainer.cuda.get_device(gpu).use()
            model.to_gpu()

        ## Set up Optimizer =========================================
        optimizer = optimizers.Adam()
        optimizer.setup(model)

        ## Set up iterator ==========================================
        train_iter = iterators.SerialIterator(train, self.batchsize)
        test_iter = chainer.iterators.SerialIterator(test, self.batchsize,
                                                 repeat=False, shuffle=False)

        ## Updater ==================================================
        updater = training.StandardUpdater(train_iter, optimizer, device=gpu)

        ## Trainer ==================================================
        trainer = training.Trainer(updater, (self.epoch, 'epoch'), out=self.outdir)

        ## Evaluator ================================================
        trainer.extend(TestModeEvaluator(test_iter, model, device=gpu))
        #trainer.extend(extensions.ExponentialShift('lr', 0.5),
        #               trigger=(25, 'epoch'))
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.snapshot(filename='snapshot'))
        trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
        trainer.extend(extensions.LogReport(log_name='log0'))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/accuracy', 'validation/main/accuracy', \
             'main/f_value_0', 'validation/main/f_value_0', 'elapsed_time']))

        trainer.extend(extensions.ProgressBar())

        trainer.run()
        serializers.save_npz(cnn_model_path, model)
        print('model saved')
        return model

    def predict(self, test_data, test_label, predictorpath):
        

        cnn = CNN(self.output_ch1, self.output_ch2, self.filter_height, \
                                     self.width, self.n_units, self.n_label)
        predictor = SClassifier(cnn, self.n_label)
        serializers.load_npz(predictorpath, predictor)


        acc, f1 = predictor(test_data, test_label)

        return acc, f1

    def extract(self, data, predictorpath):

        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                cnn = CNN(self.output_ch1, self.output_ch2, self.filter_height, \
                          self.width, self.n_units, self.n_label)
                predictor = SClassifier(cnn, self.n_label)
                serializers.load_npz(predictorpath, predictor)
                data = cnn.extract(data)

        return data