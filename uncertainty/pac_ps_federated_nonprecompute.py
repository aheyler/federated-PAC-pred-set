import os, sys
import numpy as np
import pickle
import types
import itertools
import scipy
import math
import warnings

import torch as tc

from learning import *
from uncertainty import *
import model
from uncertainty.pac_ps import PredSetConstructorFederated
from .util import *
    

def compute_k(n, e, d):
    r = 0.0
    s = 0.0

    for h in range(n + 1):
        if h == 0:
            r = n * np.log(1.0 - e)
        else:
            r += np.log(n - h + 1) - np.log(h) + np.log(e) - np.log(1.0 - e)
        s += np.exp(r)
	
        if s > d:
            if h == 0:
                raise Exception("Compute k: s > d when h is 0")
            else:
                return h - 1
    return n

class PredSetConstructor_Federated(PredSetConstructorFederated):
    def __init__(self, model, params=None, name_postfix=None):
        super().__init__(model=model, params=params, name_postfix=name_postfix)

        
    def train(self, ld, params=None):
        if params is None:
            params = self.params
            
        # Count number of validation examples
        num_val = 0
        for participant_ld in ld: 
            for _, y in participant_ld: 
                num_val += len(y)
                
        eps = params.eps
        delta = params.delta
        if self.name_postfix is None:
            self.name_postfix = ''    
        self.name_postfix = self.name_postfix + f'_n_{num_val}_eps_{eps:e}_delta_{delta:e}'
        verbose = params.verbose
        print(f"## construct a prediction set: m = {num_val}, eps = {eps:.2e}, delta = {delta:.2e}")

        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True
         

        # Solve for k*
        k = compute_k(num_val, eps, delta)
        print(f"K = {k}")
        
        # k = 0
        # binom = math.factorial(num_val) / (math.factorial(num_val - k) * math.factorial(k))
        # lhs = binom * (eps)**k * (1-eps)**(num_val - k)
        # while lhs < delta: 
        #     k += 1
        #     binom = math.factorial(num_val) / (math.factorial(num_val - k) * math.factorial(k))
        #     lhs += binom * (eps)**k * (1-eps)**(num_val - k)

        # k = max(0, k-1) # get last k before exceeding delta
        
        # neg_log_likelihoods = tc.Tensor([]).to(self.device)
        # labels = tc.Tensor([]).to(self.device)
        # for participant_ld in ld: 
        #     participant_nlls = tc.Tensor([]).to(self.device)
        #     participant_labels = tc.Tensor([]).to(self.device)
        #     for x, y in participant_ld: 
        #         x = x.to(self.device)
        #         y = y.to(self.device)
        #         x_nll = self.global_model.forward(x)
        #         participant_nlls = tc.cat((participant_nlls, x_nll), dim=0)
        #         participant_labels = tc.cat((participant_labels, y), dim=0)

        #     neg_log_likelihoods = tc.cat((neg_log_likelihoods, participant_nlls), dim=0)
        #     labels = tc.cat((labels, participant_labels), dim=0)
        #     print(f" NLLs shape = {neg_log_likelihoods.shape}")
        #     print(f" labels shape = {labels.shape}")
        # # neg_log_likelihoods = (num_val, 62)
        # # labels = (num_val,)
        
        ## save parameters
        self.mdl.n.data = tc.tensor(num_val)
        self.mdl.eps.data = tc.tensor(eps)
        self.mdl.delta.data = tc.tensor(delta)
        self.mdl.to(self.params.device)

        # Binary search on T between 0 to 1
        window_size = 1
        high = 1
        low = 0
        while window_size > 1e-5:
            # prob_thresh_T \in [0, 1]
            if low == 0 and high == 1: 
                prob_thresh_T = 0.25
            else: 
                prob_thresh_T = (low + high) / 2
            
            # compute corresponding T where the prob_thresh_T = e**(-T)
            T = - math.log(prob_thresh_T)
            print(f"T in main loop = {T}")
            self.mdl.T.data = tc.tensor(T)
            
            # Count errors
            errs = 0
            for participant_ld in ld: 
                for x, y in participant_ld:
                    x, y = to_device(x, self.params.device), to_device(y, self.params.device)
                    y_in_set = self.mdl.membership(x, y)
                    errs += sum(y_in_set == False)

            # Update window - flipped
            if errs == k: # Exact threshold found --> keep this T
                window_size = 0
            elif errs > k: # Too many errors --> search below prob_thresh_T
                high = prob_thresh_T
                window_size = high - low
            else: # Too few errors --> search above prob_thresh_T
                low = prob_thresh_T
                window_size = high - low
            print(f"Probability threshold={prob_thresh_T:.4f}, T={T:.4f}, Errors = {errs}")
            
        print(f'T_opt = {T:.8f}')


        ## save models
        self._save_model(best=True)
        self._save_model(best=False)
        print()

        return True