import param_run

"""
    alpha: outcome regularization parameter
    beta: treatment effect regularization parameter
    sigma: scaling parameter for Gaussian kernel
    rd: regularization decay parameter for label propagation
    comp: the size of dimension after dimensionality reduction by PCA
"""

""" IHDP """
param_dict = {
    'alpha':[1e-3,1e-2,1e-1, 1, 1e1, 1e2],
    'beta':[1e-3,1e-2,1e-1, 1, 1e1, 1e2],
    'sigma':[5e-3,1e-3,5e-2,1e-2,5e-1,1e-1,1,5,1e1,5e1,1e2,5e2],
    'batch_size':[32],
    'seed': list(range(1,51)),
    'rd':[0.99, 0.9],
    'comp':[2,4,8,16]
}

""" News """
param_dict = {
    'alpha':[1e-3,1e-2,1e-1, 1, 1e1, 1e2],
    'beta':[1e-3,1e-2,1e-1, 1, 1e1, 1e2],
    'sigma':[5e-3,1e-3,5e-2,1e-2,5e-1,1e-1,1,5,1e1,5e1,1e2,5e2],
    'batch_size':[32],
    'seed': list(range(1,11)),
    'tfidf':[False],
    'rd':[0.9],
    'comp':[2,4,8,16]
}


GPU=0
RATIO=0.1
EPOCH=4000
for rd in param_dict['rd']:
    for comp in param_dict['comp']:
        for alpha in param_dict['alpha']:
            for beta in param_dict['beta']:
                for sigma in param_dict['sigma']:
                    for batch in param_dict['batch_size']:
                        for seed in param_dict['seed']:
                            output = 'ihdp_proposed_'+str(RATIO)+'_alpha'+str(alpha)+'-beta'+str(beta)+'-k'+str(746)+'-weight'+str(1)+'-sigma'+str(sigma)+'-b'+str(batch)
                            target_dict = {
                                'alpha':alpha,
                                'beta': beta,
                                'sigma':sigma,
                                'batch_size':batch,
                                'semi_batch_size':batch,
                                'seed':seed,
                                'output': output,
                                'epochs':EPOCH,
                                'data':'ihdp',
                                'train_ratio': RATIO,
                                'gpu':GPU,
                                'rd':rd
                            }
                            param_run.main(target_dict)
