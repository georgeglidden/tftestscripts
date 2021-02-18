from timeit import timeit
from tensorflow.linalg import matmul,eig,svd
from tensorflow.random import uniform,shuffle
from tensorflow import gather,range,shape,GradientTape
from tensorflow.keras.layers import Dense,BatchNormalization,ReLU,Conv2D,AvgPool2D,Input
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import get as get_ls
from tensorflow.keras.optimizers import get as get_op
import os
from os import path
import sys
import glob

def do_matops(shape=(100,100)):
    A = uniform(shape)
    B = uniform(shape)
    C = matmul(A,B)
    D = matmul(B,A)
    eig(A);eig(B);eig(C);eig(D)
    svd(A);svd(B);svd(C);eig(D)

def do_prepops(shape=(10000,50,50)):
    ds = uniform(shape)
    permutation = shuffle(range(shape[0]))
    permuted_ds = gather(ds, permutation, axis=0)

def build_model_dense(architecture=None):
    if architecture == None:
        depth = int(uniform([1],2,10))
        units = [2**int(uniform([1],2,9)) for _ in range(depth)]
        activations = ['elu','exponential','linear','relu','selu','swish','tanh']
        architecture = []
        for i in range(depth):
            j = int(uniform([1],0,len(activations)))
            doNorm = int(uniform([1],0,2))
            architecture.append((units[i],activations[j],doNorm))
    inp = Input(architecture[0][0])
    x = Dense(architecture[0][0],activation=architecture[-0][1])(inp)
    for units,acti,doNorm in architecture[1:-1]:
        x = Dense(units,activation=acti)(x)
        if doNorm:
            x = BatchNormalization()(x)
    out = Dense(architecture[-1][0],activation=architecture[-1][1])(x)
    loss = 'categorical_crossentropy'
    optimizers = ['adadelta','adagrad','adam','adamax','ftrl','nadam','rmsprop','sgd']
    optimizer = optimizers[int(uniform([1],0,len(optimizers)))]
    return Model(inputs=inp,outputs=out),architecture[0][0],loss,optimizer

def save_models(models, save_dir='random_architectures', get_name=None):
    if get_name == None:
        get_name = lambda i,m: f'model{i}-{m[1]}-{m[2]}-{m[3]}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for i in range(len(models)):
        fpath = path.join(save_dir, get_name(i,models[i]))
        models[i][0].save(fpath)

def do_eval(model, insh):
    x = uniform(insh)
    y = model(x)
    return (x,y)

def do_gradops(opt,loss,model,insh,t=None):
    with GradientTape() as tape:
        y = do_eval(model, insh)[1]
        if t == None:
            t = uniform(shape(y))
        l = loss(t,y)
    grad = tape.gradient(l,model.trainable_weights)
    opt.apply_gradients(zip(grad,model.trainable_weights))

def main():
    if sys.argv[1][0] == 'g':
        models = [build_model_dense() for _ in range(int(sys.argv[2]))]
        save_models(models)
    elif sys.argv[1][0] == 'k':
        print('keras operations')
        models_dir = sys.argv[2]
        model_paths = glob.glob(path.join(models_dir, '*'))
        for i in range(len(model_paths)):
            model = load_model(model_paths[i])
            model.summary()
            insh, loss, opti = model_paths[i].split('-')[-3:]
            ls_fn = get_ls(loss)
            op_fn = get_op(opti)
            print(insh,loss,opti)
            print('inference operations')
            insh = [1,int(insh)]
            for nstr in sys.argv[3:]:
                n = int(nstr)
                print(n, timeit(lambda: do_eval(model,insh),number=n)/n)
            print('training operations')
            for nstr in sys.argv[3:]:
                n = int(nstr)
                print(n, timeit(lambda: do_gradops(op_fn,ls_fn,model,insh),number=n)/n)
    elif sys.argv[1][0] == 'm':
        print('matrix operations')
        for nstr in sys.argv[2:]:
            n = int(nstr)
            print(n, timeit(lambda: do_matops(),number=n)/n)
    elif sys.argv[1][0] == 'd':
        print('data prep operations')
        for nstr in sys.argv[2:]:
            n = int(nstr)
            print(n, timeit(lambda: do_prepops(),number=n)/n)

if __name__ == '__main__':
    main()
