import os, sys
import time
import numpy as np
import copy
import torch as tc
from torch import nn, optim
from copy import deepcopy

class BaseLearner:
    def __init__(self, mdl, params=None, name_postfix=None, local_mdl=None):
        self.params = params
        self.mdl = mdl
        self.name_postfix = name_postfix
        self.loss_fn_train = None
        self.loss_fn_val = None
        self.loss_fn_test = None
        if params:
            self.mdl_fn_best = os.path.join(params.snapshot_root, params.exp_name, 'model_params%s_best') 
            self.mdl_fn_final = os.path.join(params.snapshot_root, params.exp_name, 'model_params%s_final')
            self.mdl_fn_chkp = os.path.join(params.snapshot_root, params.exp_name, 'model_params%s_chkp')
            self.err_arrays_chkp = os.path.join(params.snapshot_root, params.exp_name)
            self.mdl.to(self.params.device)


    def _load_model(self, best=True):
        if best:
            model_fn = self.mdl_fn_best%('_'+self.name_postfix if self.name_postfix else '')
        else:
            model_fn = self.mdl_fn_final%('_'+self.name_postfix if self.name_postfix else '')
        print(model_fn)
        print(f'[{"best" if best else "final" } model is loaded] {model_fn}')
        self.mdl.load_state_dict(tc.load(model_fn))
        return model_fn

        
    def _save_model(self, best=True):
        if best:
            model_fn = self.mdl_fn_best%('_'+self.name_postfix if self.name_postfix else '')
        else:
            model_fn = self.mdl_fn_final%('_'+self.name_postfix if self.name_postfix else '')
        os.makedirs(os.path.dirname(model_fn), exist_ok=True)
        tc.save(self.mdl.state_dict(), model_fn)
        return model_fn

    
    def _check_model(self, best=True):
        if best:
            model_fn = self.mdl_fn_best%('_'+self.name_postfix if self.name_postfix else '')
        else:
            model_fn = self.mdl_fn_final%('_'+self.name_postfix if self.name_postfix else '')
        return os.path.exists(model_fn)
        
    
    def _save_chkp(self):
        model_fn = self.mdl_fn_chkp%('_'+self.name_postfix if self.name_postfix else '')
        chkp = {
            'epoch': self.i_epoch,
            'mdl_state': self.mdl.state_dict(),
            'opt_state': self.opt.state_dict(),
            'sch_state': self.scheduler.state_dict(),
            'error_val_best': self.error_val_best,
        }
        tc.save(chkp, model_fn)
        return model_fn


    def _load_chkp(self, chkp_fn):
        return tc.load(chkp_fn, map_location=tc.device('cpu'))

    
    def train(self, ld_tr, ld_val=None, ld_test=None):
        ## load a model
        if not self.params.rerun and not self.params.resume and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return
        
        self._train_begin(ld_tr, ld_val, ld_test)
        for i_epoch in range(self.epoch_init, self.params.n_epochs+1):

            # just change this to a normal training loop instead of modifying train_epoch one at a time
            # maybe start by averaging weights --> fix later to average gradient updates
            self.i_epoch = i_epoch
            self._train_epoch_begin(i_epoch)
            self._train_epoch(i_epoch, ld_tr)
            self._train_epoch_end(i_epoch, ld_val, ld_test)
        self._train_end(ld_val, ld_test)
        

    def validate(self, ld):
        return self.test(ld, mdl=self.mdl, loss_fn=self.loss_fn_val)

    
    def test(self, ld, model=None, loss_fn=None):
        model = model if model else self.mdl
        loss_fn = loss_fn if loss_fn else self.loss_fn_test
        loss_vec = []
        with tc.no_grad():
            for x, y in ld:
                loss_dict = loss_fn(x, y, model, reduction='none', device=self.params.device)
                loss_vec.append(loss_dict['loss'])
        loss_vec = tc.cat(loss_vec)
        loss = loss_vec.mean()
        return loss,
            

    def _train_begin(self, ld_tr, ld_val, ld_test):
        self.time_train_begin = time.time()
        
        ## init an optimizer
        if self.params.optimizer == "Adam":
            self.opt = optim.Adam(self.mdl.parameters(), lr=self.params.lr)
        elif self.params.optimizer == "AMSGrad":
            self.opt = optim.Adam(self.mdl.parameters(), lr=self.params.lr, amsgrad=True)
        elif self.params.optimizer == "SGD":
            self.opt = optim.SGD(self.mdl.parameters(), lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay)
        else:
            raise NotImplementedError
        
        ## init a lr scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.opt, self.params.lr_decay_epoch, self.params.lr_decay_rate)    

        ## resume training
        if self.params.resume:
            chkp = self._load_chkp(self.params.resume)
            self.epoch_init = chkp['epoch'] + 1
            self.opt.load_state_dict(chkp['opt_state'])
            self.scheduler.load_state_dict(chkp['sch_state'])
            self.mdl.load_state_dict(chkp['mdl_state'])
            self.error_val_best = chkp['error_val_best']
            self.mdl.to(self.params.device)
            print(f'## resume training from {self.params.resume}: epoch={self.epoch_init} ')
        else:
            ## init the epoch_init
            self.epoch_init = 1
        
            ## measure the initial model validation loss
            if ld_val:
                self.error_val_best, *_ = self.validate(ld_val)
            else:
                self.error_val_best = np.inf

            self._save_model(best=True)
        
    
    def _train_end(self, ld_val, ld_test):

        ## save the final model
        fn = self._save_model(best=False)
        print('## save the final model to %s'%(fn))
        
        ## load the model
        if not self.params.load_final:
            fn = self._load_model(best=True)
            print("## load the best model from %s"%(fn))

        ## print training time
        if hasattr(self, 'time_train_begin'):
            print("## training time: %f sec."%(time.time() - self.time_train_begin))
        
    
    def _train_epoch_begin(self, i_epoch):
        self.time_epoch_begin = time.time()
        

    def _train_epoch_batch_begin(self, i_epoch):
        pass

    
    def _train_epoch_batch_end(self, i_epoch):
        pass

    
    def _train_epoch(self, i_epoch, ld_tr):
        for x, y in ld_tr:
            self._train_epoch_batch_begin(i_epoch)
            self.opt.zero_grad()            
            self.loss_dict = self.loss_fn_train(x, y, lambda x: self.mdl(x, training=True), reduction='mean', device=self.params.device)
            
            if hasattr(self.mdl, "is_ode") and self.mdl.is_ode:
                nfe_forward = self.mdl.feature_layers[0].nfe
                self.mdl.feature_layers[0].nfe = 0

            self.loss_dict['loss'].backward()
            self.opt.step()

            if hasattr(self.mdl, "is_ode") and self.mdl.is_ode:
                nfe_backward = self.mdl.feature_layers[0].nfe
                self.mdl.feature_layers[0].nfe = 0

            self._train_epoch_batch_end(i_epoch)
        self.scheduler.step()


    def _train_epoch_end(self, i_epoch, ld_val, ld_test):

        msg = ''
        
        ## print loss
        for k, v in self.loss_dict.items():
            msg += '%s = %.4f, '%(k, v)

        ## test error
        if ld_test:
            error_te, *_ = self.test(ld_test)
            msg += 'error_test = %.4f, '%(error_te)
            
        ## validate the current model and save if it is the best so far
        if ld_val and (i_epoch % self.params.val_period==0):
            error_val, *_ = self.validate(ld_val)
            msg += 'error_val = %.4f (error_val_best = %.4f)'%(error_val, self.error_val_best)
            if self.error_val_best >= error_val:
                msg += ', saved'
                self._save_model(best=True)
                self.error_val_best = error_val
        elif ld_val is None:
            self._save_model(best=False)
            msg += 'saved'

        ## print the current status
        msg = '[%d/%d epoch, lr=%.2e, %.2f sec.] '%(
            i_epoch, self.params.n_epochs, 
            self.opt.param_groups[0]['lr'],
            time.time()-self.time_epoch_begin,
        ) + msg
        
        print(msg)

        ## save checkpoint
        self._save_chkp()

#################################################################################################################################

class BaseFederatedLearner(BaseLearner):
    # take in global model, and list of local models
    def __init__(self, mdl, local_mdl, params=None, name_postfix=None):
        BaseLearner.__init__(self, mdl, params=params, name_postfix=name_postfix, local_mdl=local_mdl)
        self.device = "cuda" if tc.cuda.is_available() else "cpu"
        self.global_model = mdl
        self.local_model = local_mdl
        

    def models_to_device(self): 
        self.global_model = self.global_model.to(self.device)
        self.local_model = self.local_model.to(self.device)


    def set_optimizer(self): 
        ## init an optimizer
        if self.params.optimizer == "Adam":
            return optim.Adam(self.local_model.parameters(), lr=self.params.lr)
        elif self.params.optimizer == "AMSGrad":
            return optim.Adam(self.local_model.parameters(), lr=self.params.lr, amsgrad=True)
        elif self.params.optimizer == "SGD":
            return optim.SGD(self.local_model.parameters(), lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay)
        else:
            raise NotImplementedError("No optimizer found")


    def copy_global_model(self): 
        global_state_dict = deepcopy(self.global_model.state_dict())
        self.local_model.load_state_dict(global_state_dict)
            

    def train(self, train_loader_list, val_loader_list=None, test_loader_list=None):
        print("Entered train")
        self.num_participants = len(train_loader_list)
        print(f"Number of participants: {self.num_participants}")
        
        if not self.params.rerun and not self.params.resume and self._check_model(best=False):
            if self.params.load_final: 
                self._load_model(best=False)
            else: 
                self._load_model(best=True)
            return None, None
            print("Model loading if applicable complete")

        self.time_train_begin = time.time()

        ## init a optimizer and lr scheduler
        self.opt = self.set_optimizer()
        self.scheduler = optim.lr_scheduler.StepLR(self.opt, self.params.lr_decay_epoch, self.params.lr_decay_rate)
        
        print("Set optimizer")

        # If resuming training: 
        if self.params.resume:
            print("Resuming training")
            chkp = self._load_chkp(self.params.resume)
            self.epoch_init = chkp['epoch']
            self.opt.load_state_dict(chkp['opt_state'])
            self.scheduler.load_state_dict(chkp['sch_state'])
            self.global_model.load_state_dict(chkp['mdl_state'])
            self.error_val_best = chkp['error_val_best']
            self.global_model.to(self.params.device)
            print(f'## resume training from {self.params.resume}: epoch={self.epoch_init} ')
        
        # Otherwise start training
        else: 
            print("Beginning training", flush=True)
            ## init the epoch_init
            self.epoch_init = 1
            training_errors = []
            test_errors = []
            val_errors = []
            self.models_to_device()
        
            # measure the initial model validation loss
            if val_loader_list:
                self.error_val_best, *_ = self.validate(val_loader_list)
            else:
                self.error_val_best = np.inf

            self._save_model(best=True)

            # Training loop begins
            for i_epoch in range(self.epoch_init, self.params.n_epochs + 1):
                print(f"Beginning epoch {i_epoch}", flush=True)
                self.i_epoch = i_epoch
                self.time_epoch_begin = time.time()
                self.sizes = []
                training_incorrect = 0
                
                # Federated learning: Loop through each participant
                for participant_idx in range(self.num_participants): 
                    participant_loader = train_loader_list[participant_idx]
                    
                    # Create local model that inherits weights from the global model
                    self.copy_global_model()
                    participant_errs = 0
                    participant_size = 0

                    for x, y in participant_loader:
                        participant_size += len(y)
                        x = x.to(self.params.device)
                        y = y.to(self.params.device)
                        
                        # Zero out the gradient from the optimizer
                        self.opt.zero_grad()

                        # Compute loss through network
                        self.loss_dict = self.loss_fn_train(x, y, lambda x: self.local_model(x, training=True), \
                                reduction='mean', device=self.params.device)
                        self.loss_dict['loss'].backward()
                        
                        # Step weight adjustment
                        self.opt.step()
                        
                        # Measure accuracy
                        y_pred = self.local_model(x)['yh_top']
                        participant_errs += sum(y != y_pred).item()
                    
                    training_incorrect += participant_errs
                    
                    if participant_idx % 50 == 0:
                        print(f"Participant {participant_idx} (n={participant_size}) error rate: {participant_errs/participant_size}", 
                              flush=True)
                    
                    # Keep track of participant sizes
                    self.sizes.append(participant_size)
                    
                    # Update weights to global model
                    local_model_params = deepcopy(self.local_model.state_dict())            
                    
                    if participant_idx == 0: 
                        new_global_model_params = {}
                        for key in local_model_params: 
                            new_global_model_params[key] = tc.multiply(local_model_params[key], participant_size)
                    else: 
                        for key in new_global_model_params: 
                            new_global_model_params[key] = tc.add(new_global_model_params[key], 
                                                                tc.multiply(local_model_params[key], participant_size))

                # Divide by total number of examples
                total_num_samples = sum(self.sizes)
                for key in new_global_model_params: 
                    new_global_model_params[key] = tc.div(new_global_model_params[key], total_num_samples)
                
                # Update global model
                self.global_model.load_state_dict(new_global_model_params)
                
                # Step optimizer
                self.scheduler.step()

                self._train_epoch_batch_end(i_epoch)                

                # train epoch end
                msg = ''
            
                ## print loss
                for k, v in self.loss_dict.items():
                    msg += '%s = %.4f, '%(k, v)

                ## test error
                if test_loader_list:
                    error_te, *_ = self.test(test_loader_list)
                    test_errors.append(error_te.item())
                    msg += 'error_test = %.4f, '%(error_te)
                    
                ## validate the current model and save if it is the best so far
                if val_loader_list and (i_epoch % self.params.val_period==0):
                    error_val, *_ = self.validate(val_loader_list)
                    val_errors.append(error_val.item())
                    msg += 'error_val = %.4f (error_val_best = %.4f)'%(error_val, self.error_val_best)
                    if self.error_val_best >= error_val:
                        msg += ', saved'
                        self._save_model(best=True)
                        self.error_val_best = error_val
                elif val_loader_list is None:
                    self._save_model(best=False)
                    msg += 'saved'

                ## print the current status
                msg = '[%d/%d epoch, lr=%.2e, %.2f sec.] '%(
                    i_epoch, self.params.n_epochs, 
                    self.opt.param_groups[0]['lr'],
                    time.time()-self.time_epoch_begin,
                ) + msg
                
                print(msg, flush=True)
                
                training_errors.append(training_incorrect / sum(self.sizes))
                print(f"{self.err_arrays_chkp}/training_errors.npy")
                np.save(f"{self.err_arrays_chkp}/training_errors.npy", training_errors)
                np.save(f"{self.err_arrays_chkp}/val_errors.npy", val_errors)
                np.save(f"{self.err_arrays_chkp}/test_errors.npy", test_errors)

            # train end
            ## save the final model
            fn = self._save_model(best=False)
            print('## save the final model to %s'%(fn))
            
            ## load the model
            if not self.params.load_final:
                fn = self._load_model(best=True)
                print("## load the best model from %s"%(fn))

            ## print training time
            if hasattr(self, 'time_train_begin'):
                print("## training time: %f sec."%(time.time() - self.time_train_begin))
        
        print(f"Final test error = {test_errors[-1]}")
        print(f"Final val error = {val_errors[-1]}")
        
        # Save to self.mdl 
        self.mdl = self.mdl.load_state_dict(self.global_model.state_dict())
                

    def validate(self, ld):
        return self.test(ld, mdl=self.global_model, loss_fn=self.loss_fn_val)

    def test(self, loader_list, model=None, loss_fn=None):
        model = model if model else self.global_model
        loss_fn = loss_fn if loss_fn else self.loss_fn_test
        loss_vec = []
        with tc.no_grad():
            for ld in loader_list: 
                for x, y in ld:
                    loss_dict = loss_fn(x, y, model, reduction='none', device=self.params.device)
                    loss_vec.append(loss_dict['loss'])
        loss_vec = tc.cat(loss_vec)
        loss = loss_vec.mean()
        return loss,
                
    def _save_chkp(self):
        model_fn = self.mdl_fn_chkp%('_'+self.name_postfix if self.name_postfix else '')
        chkp = {
            'epoch': self.i_epoch,
            'mdl_state': self.global_model.state_dict(),
            'opt_state': self.opt.state_dict(),
            'sch_state': self.scheduler.state_dict(),
            'error_val_best': self.error_val_best,
        }
        tc.save(chkp, model_fn)
        return model_fn

