import argparse

""" Training configuration """

parser = argparse.ArgumentParser(description='Counterfactual Propagation')
parser.add_argument('--batch_size', '-b', type=int, default=32,
                    help='input mini-batch size for training')
parser.add_argument('--semi_batch_size', '-sb', type=int, default=32,
                    help='input mini-batch size for label propagation')
parser.add_argument('--test-batch-size','-tb', type=int, default=200,
                    help='input mini-batch size for testing')
parser.add_argument('--epochs','-e', type=int, default=20,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr','-lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu','-gpu', type=int, default=0,
                    help='device id of gpu')
parser.add_argument('--seed','-s', type=int, default=1,
                    help='random seed for data indexing')
parser.add_argument('--train_seed','-ts', type=int, default=1,
                    help='random seed for parameters and training')
parser.add_argument('--verbose','-verbose', type=int, default=1,
                    help='validation interval')
parser.add_argument('--report','-r', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--tfidf','-tfidf', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--data','-d', type=str, default='ihdp',
                    help='dataset type')
parser.add_argument('--output_name','-output', type=str, default='temp.txt',
                    help='the name of output file')
parser.add_argument('--output_dir','-output_d', type=str, default='results/',
                    help='the name of output directory')
parser.add_argument('--dim','-dim', type=int, default=100,
                    help='the size of representation dimension')
parser.add_argument('--train_ratio','-ratio', type=float, default=0.1,
                    help='The proportion of labeled data')
parser.add_argument('--alpha','-alpha', type=float, default=1.,
                    help='The strength of outcome regularization')
parser.add_argument('--beta','-beta', type=float, default=1.,
                    help='The strength of treatment effect regularization')
parser.add_argument('--start_epoch','-start_epoch', type=int, default=5,
                    help='The number of first supervised learning')
parser.add_argument('--sigma','-sigma', type=float, default=1,
                    help='For Saving the current Model')
parser.add_argument('--dp','-dp', type=float, default=0.2,
                    help='The ratio of drop out')
parser.add_argument('--comp','-comp', type=int, default=4,
                    help='The size of dimension for PCA')
parser.add_argument('--reg_decay','-rd', type=float, default=0.99,
                    help='The strength of regularization decay')
parser.add_argument('--threthold','-th', type=float, default=1e-10,
                    help='Thethhold to remove pairs whose similarities are too small')
parser.add_argument('--noise','-noise', type=float, default=0.,
                    help='The magnitude of noise (for synthetic dataset)')
args = parser.parse_args()
