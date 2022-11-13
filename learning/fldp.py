# Source: https://github.com/Yangfan-Jiang/Federated-Learning-with-Differential-Privacy/blob/master/FLModel.py
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
import numpy as np
import copy
from learning import *


def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape).to(device)

class FLClient(nn.Module):
    """ Client of Federated Learning framework.
        1. Receive global model from server
        2. Perform local training (compute gradients)
        3. Return local model (gradients) to server
    """
    def __init__(self, model, data, lr, E, q, clip, sigma, device=None):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(FLClient, self).__init__()
        self.device = device
        self.torch_dataset = data
        self.data_size = len(self.torch_dataset)
        self.sigma = sigma    # DP noise level
        self.lr = lr
        self.E = E
        self.clip = clip
        self.q = q
        self.model = model.to(self.device)

    def recv(self, model_param):
        """receive global model from aggregator (server)"""
        self.model.load_state_dict(copy.deepcopy(model_param))

    def update(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.0)

        for e in range(self.E):
            # randomly select q fraction samples from data
            # according to the privacy analysis of moments accountant
            # training "Lots" are sampled by poisson sampling
            # idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]

            # MODIFY HERE
            # sampled_dataset = TensorDataset(self.torch_dataset[idx][0], self.torch_dataset[idx][1])
            # sample_data_loader = DataLoader(
            #     dataset=sampled_dataset,
            #     batch_size=self.BATCH_SIZE,
            #     shuffle=True
            # )
            
            optimizer.zero_grad()

            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            for batch_x, batch_y in self.torch_dataset:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x.float())
                loss = criterion(pred_y['fh'], batch_y)
                # loss.backward()
                # loss_dict = criterion(batch_x, batch_y, lambda x: self.model(x, training=True), \
                #                       reduction='mean', device=self.device)
                # loss_dict['loss'].backward()
                                
                # bound l2 sensitivity (gradient clipping)
                # clip each of the gradient in the "Lot"

                for i in range(loss.size()[0]):
                    loss[i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad 
                    self.model.zero_grad()
                    
            # add Gaussian noise
            for name, param in self.model.named_parameters():
                clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, self.clip, self.sigma, device=self.device)
                
            # scale back
            for name, param in self.model.named_parameters():
                clipped_grads[name] /= (self.data_size*self.q)
            
            for name, param in self.model.named_parameters():
                param.grad = clipped_grads[name]
            
            # update local model
            optimizer.step()

class FLServer(nn.Module):
    """ Server of Federated Learning
        1. Receive model (or gradients) from clients
        2. Aggregate local models (or gradients)
        3. Compute global model, broadcast global model to clients
    """
    def __init__(self, fl_param, test_dataset):
        super(FLServer, self).__init__()
        self.device = fl_param['device']
        self.client_num = fl_param['client_num']
        self.C = fl_param['C']      # (float) C in [0, 1]
        self.epochs = fl_param['tot_epochs']  # total number of global iterations (communication rounds)
        self.test_dataset = test_dataset
        self.clip = fl_param['clip']
        # for sample in fl_param['data'][self.client_num:]:
        #     self.data += [torch.tensor(sample[0]).to(self.device)]    # test set
        #     self.target += [torch.tensor(sample[1]).to(self.device)]  # target label

        self.input_size = len(test_dataset)
        self.lr = fl_param['lr']
        
        # print("FL Server model type")
        # print(type(fl_param['model']))
        
        # compute noise using moments accountant
        self.sigma = compute_noise(1, fl_param['q'], fl_param['eps'], fl_param['E']*fl_param['tot_epochs'], fl_param['delta'], 1e-5)
        
        self.clients = [FLClient(fl_param['model'],
                                #  fl_param['output_size'],
                                 fl_param['data'][i],
                                 fl_param['lr'],
                                 fl_param['E'],
                                 fl_param['q'],
                                 self.clip,
                                 self.sigma,
                                 self.device)
                        for i in range(self.client_num)]
        
        self.global_model = fl_param['model'].to(self.device)
        self.weight = np.array([client.data_size * 1.0 for client in self.clients])
        self.broadcast(self.global_model.state_dict())

    def aggregated(self, idxs_users):
        """FedAvg"""
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(model_par):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            for name in new_par:
                # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
                new_par[name] += par[name] * (w / self.C)
        self.global_model.load_state_dict(copy.deepcopy(new_par))
        return self.global_model.state_dict().copy()

    def broadcast(self, new_par):
        """Send aggregated model to all clients"""
        for client in self.clients:
            client.recv(new_par.copy())

    def test_acc(self):
        self.global_model.eval()
        correct = 0
        tot_sample = 0
        for participant_loader in self.test_dataset: 
            for x_test, y_test in participant_loader:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                t_pred_y = self.global_model(x_test)["fh"]
                _, predicted = torch.max(t_pred_y, 1)
                correct += (predicted == y_test).sum().item()
                tot_sample += y_test.size(0)
            acc = correct / tot_sample
        return acc

    def global_update(self):
        idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
        for idx in idxs_users:
            self.clients[idx].update()
        self.broadcast(self.aggregated(idxs_users))
        acc = self.test_acc()
        torch.cuda.empty_cache()
        return acc

    def set_lr(self, lr):
        for c in self.clients:
            c.lr = lr