import os, sys
import argparse
import warnings
import numpy as np
import math
import pickle
import types
import torch as tc
import util
import data
import model
import learning
import uncertainty
import matplotlib.pyplot as plt

    
def main(args):

    ## init datasets
    print("## init datasets: %s"%(args.data.src))    
    ds = getattr(data, args.data.src)(args.data) # Executes: data.FEMNIST(args.data)
    print()

    ## init a model
    print("## init models: %s"%(args.model.base))
    if 'FNN' in args.model.base or 'Linear' in args.model.base:
        mdl = getattr(model, args.model.base)(n_in=28*28, n_out=args.data.n_labels, path_pretrained=args.model.path_pretrained) 
        local_mdl = getattr(model, args.model.base)(n_in=28*28, n_out=args.data.n_labels, path_pretrained=args.model.path_pretrained) 
    elif 'ResNet' in args.model.base:
        mdl = getattr(model, args.model.base)(n_labels=args.data.n_labels, path_pretrained=args.model.path_pretrained)
        local_mdl = getattr(model, args.model.base)(n_labels=args.data.n_labels, path_pretrained=args.model.path_pretrained)
    elif 'OdeNet' in args.model.base: 
        mdl = getattr(model, args.model.base)(tol=1e-3, adjoint=False, downsampling_method='conv', \
            n_epochs=160, data_aug=True, lr=0.1, batch_size=128, test_batch_size=1000, \
                save='./experiment1', debug='store_true', gpu=0)
    else:
        raise NotImplementedError
        
    if args.multi_gpus:
        assert(not args.cpu)
        mdl = tc.nn.DataParallel(mdl).cuda()
    print()

    ## learn the model
    if args.train.federated: 
        l = learning.ClsFederatedLearner(mdl, local_mdl=local_mdl, params=args.train) 
        print("Created federated learner")
    else: 
        l = learning.ClsLearner(mdl, args.train)
    
    if args.model.path_pretrained is None:
        print("## train...")
        l.train(ds.train, ds.val, ds.test)
        
        print("## test...")
        l.test(ds.test, ld_name=args.data.src, verbose=True)

        print()
    else: 
        mdl.load_state_dict(tc.load(args.model.path_pretrained))
        print(f"Loaded model from {args.model.path_pretrained}")

    ## prediction set estimation
    print("## prediction set estimation")
    if args.train_ps.method == 'pac_predset_CP':
        mdl_ps = model.PredSetCls(mdl)
        l = uncertainty.PredSetConstructor_CP(mdl_ps, args.train_ps)
    elif args.train_ps.method == 'pac_predset_federated': 
        print("Begin federated PS construction")
        # mdl_ps = model.PredSetFederatedCls(mdl, eps=args.train_ps.eps, delta=args.train_ps.delta, n=args.train_ps.n)
        l = uncertainty.PredSetConstructor_Federated(mdl, args.train_ps)
    else:
        raise NotImplementedError
    l.train(ds.val)
    l.test(ds.test) #, ld_name=f'test datasets', verbose=True)
    
    
    # plt.figure()
    # plt.plot(test_errors)
    # plt.plot(val_errors)
    # plt.xlabel("Epoch")
    # plt.ylabel("Classification Error")
    # plt.title("Model Training")
    # plt.legend(["Test Error", "Validation Error"])
    # plt.savefig("snapshots/test_val_errors_during_training_oct_9.png")
    # plt.show()

    
if __name__ == '__main__':
    
    ## init a parser
    parser = argparse.ArgumentParser(description='PAC Prediction Set')

    ## meta args
    parser.add_argument('--exp_name', type=str, default="opacus_fmnist")
    parser.add_argument('--snapshot_root', type=str, default='snapshots')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--multi_gpus', action='store_true')

    ## data args
    parser.add_argument('--data.batch_size', type=int, default=16)
    parser.add_argument('--data.n_workers', type=int, default=8)
    parser.add_argument('--data.src', type=str, default='FEMNIST')
    parser.add_argument('--data.num_participants', type=int)
    parser.add_argument('--data.preselected_participants', type=str) #, default="/home/aheyler/PAC-pred-set/snapshots/opacus_v3.2/participants_arr.npy")
    parser.add_argument('--data.in_dim', type=str, default=28*28) 
    parser.add_argument('--data.n_labels', type=int, default=62) #10
    parser.add_argument('--data.seed', type=int, default=42)
    parser.add_argument('--data.ps_experiment_frac', type=float, default=1.0)
    
    ## model args
    parser.add_argument('--model.base', type=str, default='ResNet18') 
    parser.add_argument('--model.path_pretrained', type=str, default="/home/aheyler/PAC-pred-set/snapshots/non-dp-fl/latest_model_copy/model_params_best copy") #, default="/home/aheyler/PAC-pred-set/snapshots/exp_dpfl_dev/model_params_best")

    ## train args
    parser.add_argument('--train.n_labels', type=int, default=62)
    parser.add_argument('--train.federated', type=bool, default=True)
    parser.add_argument('--train.rerun', action='store_true')
    parser.add_argument('--train.load_final', action='store_true')
    parser.add_argument('--train.resume', type=str) #, default="/home/aheyler/PAC-pred-set/snapshots/opacus_fmnist_all_participants/model_params_best")
    parser.add_argument('--train.method', type=str, default='src')
    parser.add_argument('--train.optimizer', type=str, default='SGD')
    parser.add_argument('--train.n_epochs', type=int, default=200)
    parser.add_argument('--train.lr', type=float, default=0.001) 
    parser.add_argument('--train.momentum', type=float, default=0.9)
    parser.add_argument('--train.weight_decay', type=float, default=0.00001)
    parser.add_argument('--train.lr_decay_epoch', type=int, default=20)
    parser.add_argument('--train.lr_decay_rate', type=float, default=0.75)
    parser.add_argument('--train.val_period', type=int, default=1)
    ## dp train args
    parser.add_argument('--train.dp', type=bool, default=False) # clipping parameter during local updates 
    parser.add_argument('--train.clip', type=float, default=1.0) # clipping parameter during local updates 
    parser.add_argument('--train.client_sampling_rate', type=float, default=0.5) # fraction of the clients to sample for training in each federation round
    parser.add_argument('--train.E', type=float, default=1) # number of training epochs during each federation round
    parser.add_argument('--train.epsilon', type=float, default=8.0) # DP epsilon parameter
    parser.add_argument('--train.delta', type=float, default=0.001) # DP delta parameter
    # parser.add_argument('--train.q', type=float, default=1.0) # Local sampling rate of samples at each node

    ## uncertainty estimation args
    parser.add_argument('--train_ps.method', type=str, default='pac_predset_federated')
    parser.add_argument('--train_ps.rerun', action='store_true')
    parser.add_argument('--train_ps.load_final', action='store_true')
    parser.add_argument('--train_ps.verbose', type=bool, default=True)
    parser.add_argument('--train_ps.binary_search', action='store_true')
    parser.add_argument('--train_ps.bnd_type', type=str, default='direct')
    parser.add_argument('--train_ps.T_step', type=float, default=1e-7) 
    parser.add_argument('--train_ps.T_end', type=float, default=np.inf)
    parser.add_argument('--train_ps.eps_tol', type=float, default=1.25)
    parser.add_argument('--train_ps.n', type=float, default=5000)
    parser.add_argument('--train_ps.eps', type=float, default=0.01)
    parser.add_argument('--train_ps.delta', type=float, default=0.0001)
            
    args = parser.parse_args()
    args = util.to_tree_namespace(args)
    args.device = tc.device('cpu') if args.cpu else tc.device('cuda:0')
    args = util.propagate_args(args, 'device')
    args = util.propagate_args(args, 'exp_name')
    args = util.propagate_args(args, 'snapshot_root')

    ## setup logger
    os.makedirs(os.path.join(args.snapshot_root, args.exp_name), exist_ok=True)
    sys.stdout = util.Logger(os.path.join(args.snapshot_root, args.exp_name, 'out'))    

    ## print args
    util.print_args(args)

    ## run
    main(args)
