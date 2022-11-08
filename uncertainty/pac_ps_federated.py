import os
import numpy as np
import torch as tc
from learning import *
from uncertainty import *
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
    def __init__(self, model, params=None, name_postfix=None, save_logits=False):
        super().__init__(model=model, params=params, name_postfix=name_postfix)
        self.model_inf = model
        self.save_logits = save_logits
        
    def train(self, ld, params=None):
        if params is None:
            params = self.params
            
        # Count number of validation examples
        self.n = 0
        for participant_ld in ld: 
            for _, y in participant_ld: 
                self.n += len(y)
                
        self.eps = params.eps
        self.delta = params.delta
        if self.name_postfix is None:
            self.name_postfix = ''    
        self.name_postfix = self.name_postfix + f'_n_{self.n}_eps_{self.eps:e}_delta_{self.delta:e}'         

        # Solve for k*
        k = compute_k(self.n, self.eps, self.delta)
        
        nlls = tc.Tensor([]).to(self.device)
        logits = tc.Tensor([]).to(self.device)
        labels = tc.Tensor([]).to(self.device)
        for participant_ld in ld: 
            participant_logits = tc.Tensor([]).to(self.device)
            participant_nlls = tc.Tensor([]).to(self.device)
            participant_labels = tc.Tensor([]).to(self.device)
            for x, y in participant_ld: 
                x = x.to(self.device)
                y = y.to(self.device)
                with tc.no_grad(): 
                    x_logits = self.model_inf.forward(x)['ph']
                    x_nll = - x_logits.log()
                participant_logits = tc.cat((participant_logits, x_logits), dim=0)
                participant_nlls = tc.cat((participant_nlls, x_nll), dim=0)
                participant_labels = tc.cat((participant_labels, y), dim=0)

            logits = tc.cat((logits, participant_logits), dim=0)
            nlls = tc.cat((nlls, participant_nlls), dim=0)
            labels = tc.cat((labels, participant_labels), dim=0).type(tc.int64)
        # print(f" NLLs shape = {nlls.shape}")
        # print(f" labels shape = {labels.shape}")
        # neg_log_likelihoods = (num_val, 62)
        # labels = (num_val,)
        
        # Debugging - save logits of labels
        if self.save_logits: 
            logits_y = np.array([])
            for i in range(len(labels)): 
                val = logits[i][labels[i]].item()
                logits_y = np.append(logits_y, val)
            print(logits_y.shape)
            path_npy = os.path.join("/home/aheyler/snapshots/latest_femnist_ps_v4_nov1", "logits_val_arr.npy")
            print(path_npy)
            np.save(path_npy, logits_y)
        
        else: 
            print(f"\n## Training dataset results for n={len(labels)}, eps={self.eps}, delta={self.delta}, K={k}:")

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
                
                # Count errors
                set = nlls <= T
                membership = set.gather(1, labels.view(-1, 1)).squeeze(1)
                errs = sum(membership == False)
                
                # Evaluate sizes
                sizes = set.sum(1).float()
                min = tc.min(sizes).item()
                q1 = tc.quantile(sizes, 0.25, interpolation='nearest').item()
                med = tc.median(sizes).item()
                q3 = tc.quantile(sizes, 0.75, interpolation='nearest').item()
                max = tc.max(sizes).item()
                mean = tc.mean(sizes).item()
                
                # Update window - flipped
                if errs == k: # Exact threshold found --> keep this T
                    window_size = 0
                elif errs > k: # Too many errors --> search below prob_thresh_T
                    high = prob_thresh_T
                    window_size = high - low
                else: # Too few errors --> search above prob_thresh_T
                    low = prob_thresh_T
                    window_size = high - low
                print(f"Probability threshold={prob_thresh_T:.4f}, T={T:.4f}, Error rate = {errs/len(labels):.4f}, Errors = {errs}, Size = {[min, q1, med, q3, max]}, Mean size = {mean:.4f}")
                
            self.T_opt = T
            self.prob_thresh_opt = prob_thresh_T

        return True
    
    def test(self, ld, params=None):
        logits = tc.Tensor([]).to(self.device)
        labels = tc.Tensor([]).to(self.device)
        for participant_ld in ld: 
            participant_logits = tc.Tensor([]).to(self.device)
            participant_labels = tc.Tensor([]).to(self.device)
            for x, y in participant_ld: 
                x = x.to(self.device)
                y = y.to(self.device)
                with tc.no_grad(): 
                    x_logits = self.model_inf.forward(x)['ph']
                participant_logits = tc.cat((participant_logits, x_logits), dim=0)
                participant_labels = tc.cat((participant_labels, y), dim=0)

            logits = tc.cat((logits, participant_logits), dim=0)
            labels = tc.cat((labels, participant_labels), dim=0).type(tc.int64)
        # print(f" logits shape = {logits.shape}")
        # print(f" labels shape = {labels.shape}")
        
        if self.save_logits: 
            logits_y = np.array([])
            for i in range(len(labels)): 
                val = logits[i][labels[i]].item()
                logits_y = np.append(logits_y, val)
            print(logits_y.shape)
            path_npy = os.path.join("/home/aheyler/snapshots/latest_femnist_ps_v4_nov1", "logits_test_arr.npy")
            print(path_npy)
            np.save(path_npy, logits_y)
        
        else: 
            set = logits > self.prob_thresh_opt
            membership = set.gather(1, labels.view(-1, 1)).squeeze(1)
            errs = sum(membership == False)
            
            # Evaluate sizes
            sizes = set.sum(1).float()
            min = tc.min(sizes).item()
            q1 = tc.quantile(sizes, 0.25, interpolation='nearest').item()
            med = tc.median(sizes).item()
            q3 = tc.quantile(sizes, 0.75, interpolation='nearest').item()
            max = tc.max(sizes).item()
            mean = tc.mean(sizes).item()
            
            print(f"\n## Test dataset results for n={len(labels)}, eps={self.eps}, delta={self.delta}:")
            print(f"Probability threshold={self.prob_thresh_opt:.4f}, T={self.T_opt:.4f}, Error rate = {errs/len(labels):.4f}, Errors = {errs}, Size = {[min, q1, med, q3, max]}, Mean size = {mean:.4f}")
        
        return True
        
      
# non-log space k computation  
# k = 0
# binom = math.factorial(num_val) / (math.factorial(num_val - k) * math.factorial(k))
# lhs = binom * (eps)**k * (1-eps)**(num_val - k)
# while lhs < delta: 
#     k += 1
#     binom = math.factorial(num_val) / (math.factorial(num_val - k) * math.factorial(k))
#     lhs += binom * (eps)**k * (1-eps)**(num_val - k)
# k = max(0, k-1) # get last k before exceeding delta