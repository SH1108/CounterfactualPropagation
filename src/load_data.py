import numpy as np
import pandas as pd
from utils import *
from torchvision import datasets
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.stats import bernoulli
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import rbf_kernel

class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self._load()

    def _load(self):

        print('----- Loading data -----')

        if self.args.data == 'ihdp':

            """ IHDP DATA LOAD """
            data1 = np.load('../data/ihdp_npci_1-1000.train.npz')
            data2 = np.load('../data/ihdp_npci_1-1000.test.npz')
            idx = self.args.seed-1

            X = np.concatenate((data1['x'][:,:,idx], data2['x'][:,:,idx]))
            pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=self.args.comp))])
            X_neighbor = pipeline.fit_transform(X)
            Y = np.concatenate((data1['yf'][:,idx], data2['yf'][:,idx]))
            T = np.concatenate((data1['t'][:,idx], data2['t'][:,idx]))
            np.random.seed(self.args.seed)
            Y_treat = np.concatenate((data1['mu1'][:,idx], data2['mu1'][:,idx]))
            Y_cont = np.concatenate((data1['mu0'][:,idx], data2['mu0'][:,idx]))
            noise1 = self.args.noise*np.random.normal(0,1,size=len(X))
            noise0 = self.args.noise*np.random.normal(0,1,size=len(X))
            T_ = (-(T-1))
            Y = (Y_treat+noise1)*T + (Y_cont+noise0)*T_

            n_train = int(len(X)*self.args.train_ratio)
            indices = np.arange(len(X))
            train_index = indices[:n_train]
            val_index = indices[298:373]
            test_index = indices[373:747]

        elif self.args.data == 'news':

            """ NEWS DATA LOAD """
            x1 = pd.read_csv('../data/csv/topic_doc_mean_n5000_k3477_seed_'+str(self.args.seed)+'.csv.x')
            y1 = pd.read_csv('../data/csv/topic_doc_mean_n5000_k3477_seed_'+str(self.args.seed)+'.csv.y', header=None)
            X = np.zeros((5000, 3477))
            for doc_id, word, freq in zip(tqdm(x1['5000']), x1['3477'], x1['0']):
                X[doc_id-1][word-1] = freq

            T = y1[0].values
            Y = y1[1].values
            Y_treat = y1[4].values
            Y_cont = y1[3].values

            noise1 = self.args.noise*np.random.normal(0,1,size=len(X))
            noise0 = self.args.noise*np.random.normal(0,1,size=len(X))
            T_ = (-(T-1))
            Y = (Y_treat+noise1)*T + (Y_cont+noise0)*T_

            n_train = int(len(X)*self.args.train_ratio)
            indices = np.arange(len(X))
            train_index = indices[:n_train]
            val_index = indices[500:1000]
            test_index = indices[1000:]


            """ Use tfidf or not """
            if self.args.tfidf:
                transformer = TfidfTransformer()
                X_neighbor = transformer.fit_transform(X)
                X_neighbor = X_neighbor.toarray()
            else:
                X_neighbor = X
            pca = PCA(n_components=self.args.comp)
            X_neighbor = pca.fit_transform(X_neighbor)

        print('n_train: {}'.format(n_train))

        X_train, X_val, X_test = get_train_val_test(X, train_index, val_index, test_index)
        T_train, T_val, T_tes = get_train_val_test(T, train_index, val_index,  test_index)
        Y_train, Y_val, Y_test = get_train_val_test(Y, train_index, val_index, test_index)
        Y_treat_train, Y_treat_val, Y_treat_test = get_train_val_test(Y_treat, train_index, val_index, test_index)
        Y_cont_train, Y_cont_val, Y_cont_test = get_train_val_test(Y_cont, train_index, val_index, test_index)

        self.treat_std = Y_train[T_train==1].std()
        self.cont_std = Y_train[T_train==0].std()

        """ Gaussian kernel """
        W = rbf_kernel(X_neighbor, X_neighbor, gamma=1/self.args.sigma)
        W = torch.Tensor(W)
        W[torch.arange(len(W)), torch.arange(len(W))] = 0

        """ Remove small similarity """
        W[W<self.args.threthold]=0

        train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.from_numpy(train_index), torch.Tensor(T_train), torch.Tensor(Y_train))
        val_data = torch.utils.data.TensorDataset(torch.Tensor(X_val), torch.from_numpy(val_index), torch.Tensor(T_val), torch.Tensor(Y_val))
        test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.from_numpy(test_index), torch.Tensor(Y_treat_test),torch.Tensor(Y_cont_test))

        in_data = torch.utils.data.TensorDataset(torch.Tensor(np.concatenate((X_train, X_val))), torch.cat((torch.from_numpy(train_index),torch.from_numpy(val_index))),
                torch.Tensor(np.concatenate((Y_treat_train, Y_treat_val))),
                torch.Tensor(np.concatenate((Y_cont_train, Y_cont_val)))
                )

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, worker_init_fn=np.random.seed(self.args.seed))
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.args.test_batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.args.test_batch_size, shuffle=False)
        self.in_loader = torch.utils.data.DataLoader(in_data, batch_size=self.args.test_batch_size, shuffle=False)

        self.W = W
        self.in_dim = len(X[0])
        self.X = X

        self.Y_train = Y_train
        self.T_train = T_train


        print('----- Finished loading data -----')
