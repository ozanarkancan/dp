import theano
import theano.tensor as T
import numpy as np, h5py
import argparse
from layers import *
from optim import *
import time
import cPickle
import os
from dnet import *
from scipy.io import loadmat

def get_arg_parser():
    parser = argparse.ArgumentParser(prog="experiment")
    parser.add_argument("--data_path", default="/home/can/data/dp/archybrid_conllWSJToken_wikipedia2MUNK-100_fv021a_xy.mat", help="data path")
    parser.add_argument("--minibatch", default=128, type=int, help="minibatch size")
    parser.add_argument("--opt", default="sgd", help="optimization type sgd, rmsprop, adagrad, adam")
    parser.add_argument("--reg", default="dropout", help="regularization l1, l2, dropout")
    parser.add_argument("--hidden", default=[20000], nargs='+', type=int, help="number of units in hidden layer(s)")
    parser.add_argument("--epoch", default=1, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.002, type=float, help="learning rate")
    parser.add_argument("--patience", default=25, type=int, help="stopping criteria")
    parser.add_argument("--drate", default=[0.5, 0.5], nargs='+', type=float, help="dropout rate")
    parser.add_argument("--exp", default="dp", help="dp or ner")
    
    return parser

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    
    shared_x = theano.shared(
        np.asarray(data_x, dtype=theano.config.floatX),
            borrow=True
            )
    
    shared_y = theano.shared(
        np.asarray(data_y.reshape(data_y.shape[0], ), dtype=theano.config.floatX),
            borrow=True
        )
    return shared_x, T.cast(shared_y, 'int32')

def save(params):
    filename = os.path.realpath('.') + "/dp/save/net.save"
    f = file(filename, 'wb')
    for p in params:
        cPickle.dump(p.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def load(params):
    filename = os.path.realpath('.') + "/dp/save/net.save"
    f = file(filename, 'rb')
    for p in params:
        p.set_value(cPickle.load(f))
    f.close()

if __name__ == "__main__":
    parser = get_arg_parser()
    
    args = vars(parser.parse_args())
    data_path = args['data_path']
    ms = args['minibatch']
    opt = args['opt']
    reg = args['reg']
    hidden = args['hidden']
    epochs = args['epoch']
    drates = args['drate']
    
    print "Parameters:"
    print args
    
    print "...loading data"
    if args['exp'] == 'dp':
        f = h5py.File(data_path, 'r')
        x_trn = f.get('x_trn')
        x_trn = np.array(x_trn)
        y_trn = f.get('y_trn')
        y_trn = np.array(y_trn) - 1
        x_dev = f.get('x_dev')
        x_dev = np.array(x_dev)
        y_dev = f.get('y_dev')
        y_dev = np.array(y_dev) - 1
    else:
        f = loadmat(data_path + 'train.y2,y1,E,C.mat')
        x_trn = f.get('X')
        x_trn = np.array(x_trn)
        x_trn = x_trn.T
        y_trn = f.get('y')
        y_trn = np.array(y_trn) - 1
        y_trn = y_trn.T
        
        f = loadmat(data_path + 'testa.y2,y1,E,C.mat')
        x_dev = f.get('X')
        x_dev = np.array(x_dev)
        x_dev = x_dev.T
        y_dev = f.get('y')
        y_dev = np.array(y_dev) - 1
        y_dev = y_dev.T
        
        f = loadmat(data_path + 'testb.y2,y1,E,C.mat')
        x_tst = f.get('X')
        x_tst = np.array(x_tst)
        x_tst = x_tst.T
        y_tst = f.get('y')
        y_tst = np.array(y_tst) - 1
        y_tst = y_tst.T
        

        print x_trn.shape
        print y_trn.shape
        print x_dev.shape
        print y_dev.shape
    
    num_classes = len(np.unique(y_dev))
    x = T.matrix('x')
    y = T.ivector('y')
    
    dnet = DNET(num_classes)
    dnet.add_input_layer(x, dropout_rate=drates[0])
    dnet.add_hidden_layer(x_trn.shape[1], hidden[0], dropout_rate=drates[1])
    
    for i in range(len(hidden) - 1):
        dnet.add_hidden_layer(hidden[i], hidden[i + 1], dropout_rate=drates[i + 2])
    
    dnet.connect_output()
    params = dnet.get_params()
    
    d_loss = dnet.o_layer.d_loss(y)
    d_error = dnet.o_layer.d_error(y)
    loss = dnet.o_layer.loss(y)
    error = dnet.o_layer.error(y)
    
    print "...building the model"
    
    test_model = theano.function(
        inputs = [x, y],
        outputs = [error, loss],
        allow_input_downcast=True
        )
    
    grads = T.grad(d_loss, params)
    
    if opt == "rmsprop":
        updates = rmsprop(params, grads, lr = args["lr"])
    elif opt == "adagrad":
        updates = adagrad(params, grads, lr = args["lr"])
    elif opt == "adam":
        updates = adam(params, grads)
    else:
        updates = sgd(params, grads, lr = args["lr"])
    
    train_model = theano.function(
        inputs = [x, y],
        outputs = [d_error, d_loss],
        updates = updates,
        allow_input_downcast=True
        )
    
    n_train_batches = x_trn.shape[0] / ms
    n_valid_batches = x_dev.shape[0] / ms

    print "...training the model"
    
    best_epoch = 0
    best_val_err = 1
    
    nonimprovement = 0
    save(params)
    
    for i in xrange(epochs):
        start = time.time()
        
        trn_acc_errs = []
        trn_losses = []
        
        errs = [train_model(x_trn[index * ms : (index + 1) * ms, :],\
            y_trn[index * ms : (index + 1) * ms, :].reshape(ms, )) for index in xrange(n_train_batches)]
        
        for e in errs:
            trn_acc_errs.append(e[0])
            trn_losses.append(e[1])
        
        avg_trn_err = np.mean(trn_acc_errs)
        avg_trn_loss = np.mean(trn_losses)
        
        errs = [test_model(x_dev[index * ms : (index + 1) * ms, :],\
            y_dev[index * ms : (index + 1) * ms, :].reshape(ms, )) for index in xrange(n_valid_batches)]
        
        dev_acc_errs = []
        dev_losses = []
        
        for e in errs:
            dev_acc_errs.append(e[0])
            dev_losses.append(e[1])
        
        avg_dev_err = np.mean(dev_acc_errs)
        avg_dev_loss = np.mean(dev_losses)

        if args["exp"] == "ner":
            n_tst_batches = x_tst.shape[0] / ms
            errs = [test_model(x_tst[index * ms : (index + 1) * ms, :],\
                y_tst[index * ms : (index + 1) * ms, :].reshape(ms, )) for index in xrange(n_tst_batches)]
            tst_acc_errs = []
            tst_losses = []
            for e in errs:
                tst_acc_errs.append(e[0])
                tst_losses.append(e[1])
            
            avg_tst_err = np.mean(tst_acc_errs)
            avg_tst_loss = np.mean(tst_losses)
        
        end = time.time()
        
        if avg_dev_err < best_val_err:
            best_val_err = avg_dev_err
            best_epoch = i + 1
            nonimprovement = 0
            save(params)
        else:
            nonimprovement += 1
        print "Epoch: %i\ntrain loss: %f\ntrain error: %f\ndev loss: %f\ndev error: %f\nRunning Time: %f seconds\n" % (i + 1, avg_trn_loss, avg_trn_err, avg_dev_loss, avg_dev_err, end - start)
        if args["exp"] == "ner":
            print "tst loss: %f\ntst error: %f\n" % (avg_tst_loss, avg_tst_err)
        
        if nonimprovement == args["patience"]:
            break
    print "Best model at epoch: %i with dev error: %f" % (best_epoch, best_val_err)
