import argparse
import torch
import torch.optim as optim
from model import Model
import numpy as np
from trainer import Trainer
from config import args
from load_data import DataLoader
import os
import random

def main(params):

    """ param config for param_search """
    args.alpha = params['alpha']
    args.beta = params['beta']
    args.sigma = params['sigma']
    args.batch_size = params['batch_size']
    args.semi_batchsize = params['semi_batch_size']
    args.seed = params['seed']
    args.output_name = params['output']
    args.epochs = params['epochs']
    args.data = params['data']
    args.train_ratio = params['train_ratio']
    args.gpu = params['gpu']
    if args.data == 'news':
        args.tfidf = params['tfidf']

    args.reg_decay = params['rd']

    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    random.seed(args.train_seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(args.gpu if use_cuda else "cpu")
    print('Seed: {}'.format(args.seed))
    print('Data type: {}'.format(args.data))

    Data = DataLoader(args=args)
    train_loader = Data.train_loader
    val_loader = Data.val_loader
    in_loader = Data.in_loader
    test_loader = Data.test_loader

    model = Model(stds=[Data.treat_std, Data.cont_std],p=args.dp, in_dim=Data.in_dim, alpha=args.alpha, beta=args.beta, features=torch.Tensor(Data.X), device=device, out_dim=args.dim, W=Data.W).to(device)
    weight_decay=1e-8
    clipping_value = 1
    torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    trainer = Trainer(model=model, device=device, args=args, optimizer=optimizer, train_loader=train_loader, validation_loader=val_loader, in_loader=in_loader, test_loader=test_loader, D=Data)
    trainer.run()

    print('Min (out, in): {}, {}'.format(np.min(trainer.test_losses), np.min(trainer.in_losses)))
    print('Validated min (out, in): {}, {}'.format(trainer.test_losses[np.argsort(np.array(trainer.val_losses))[0]],trainer.in_losses[np.argsort(np.array(trainer.val_losses))[0]]))

    print('T mse: {}'.format(trainer.t_mse_losses[np.argsort(np.array(trainer.val_losses))[0]]))
    print('C mse: {}'.format(trainer.c_mse_losses[np.argsort(np.array(trainer.val_losses))[0]]))

    print('epoch: {}'.format(np.argsort(np.array(trainer.val_losses))[0]+1))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    f = open(args.output_dir+args.output_name,'a')
    f.write(str(float(trainer.test_losses[np.argsort(np.array(trainer.val_losses))[0]]))+',')
    f.write(str(float(trainer.in_losses[np.argsort(np.array(trainer.val_losses))[0]]))+',')
    f.write(str(float(np.min(np.array(trainer.val_losses))))+'\n')
    f.close()

if __name__ == '__main__':
    main(params)
