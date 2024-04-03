import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from collections import OrderedDict
from maml.utils import update_parameters, tensors_to_device, compute_accuracy
import copy
from utils import load_clusters, select_index
import random
import math

__all__ = ['ModelAgnosticMetaLearning', 'MAML']


class ModelAgnosticMetaLearning(object):

    def __init__(self, model, optimizer=None, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=nn.CrossEntropyLoss(), device=None,
                 dataloader=None):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        # add evaluation model
        self.eval_model = copy.deepcopy(model)
        self.eval_model = self.eval_model.to(device=device)
        # load clusters
        c_list, c_centers, ind_sorted = load_clusters(dataloader)
        self.c_list, self.c_centers, self.ind_sorted = c_list, c_centers, ind_sorted

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                dtype=param.dtype, device=self.device,
                requires_grad=learn_step_size)) for (name, param)
                in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
                if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                    for group in self.optimizer.param_groups])

    def get_outer_loss(self, batch,Train=True):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        test_targets = batch['test'][1]
        num_tasks = test_targets.size(0)
        is_classification_task = (not test_targets.dtype.is_floating_point)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }
        if is_classification_task:
            results.update({
                'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
                'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
            })

        mean_outer_loss = torch.tensor(0., device=self.device)
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(batch['train'][0],batch['train'][1], batch['test'][0],batch['test'][1])):
            #print(train_inputs.size(0))
            params, adaptation_results = self.adapt(train_inputs, train_targets, self.model,
                is_classification_task=is_classification_task,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size, first_order=self.first_order)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            if is_classification_task:
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

            with torch.set_grad_enabled(self.model.training):
                if Train==True:
                    test_inputs=self.RCL(test_inputs)
 
                else:
                    test_inputs=test_inputs
                test_logits = self.model(test_inputs, params=params)
                #print(test_logits.size(0),test_logits.size(1))
                outer_loss = self.loss_function(test_logits, test_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            if is_classification_task:
                results['accuracies_after'][task_id] = compute_accuracy(
                    test_logits, test_targets)

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results,test_logits,test_targets,test_inputs
    def RCL(self, X_n):  
        N, C, W = X_n.size()
        p = np.random.rand()
        K = [1, 3, 5, 7, 11, 15]
        if p > 0.5:
            k = K[np.random.randint(0, len(K))]
            Conv = torch.nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=k//2, bias=False)
            torch.nn.init.kaiming_normal_(Conv.weight)
            X_n = Conv(X_n.reshape(-1, C, W)).reshape(N, C,  W)
        return X_n.detach()
    def adapt(self, inputs, targets, model, is_classification_task=None,
              num_adaptation_steps=1, step_size=0.1, first_order=False):
        if is_classification_task is None:
            is_classification_task = (not targets.dtype.is_floating_point)
        params = None

        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}
        


        for step in range(num_adaptation_steps):
            #inputs=self.RCL(inputs)
            #print(inputs.size(0),inputs.size(1),inputs.size(2))
            logits = model(inputs, params=params)
            #print('label:',targets.size(0))
            #print('label:',targets)
            #print('123:',logits.size(0),logits.size(1))
            #print('123:',logits)
            
           
            
            inner_loss = self.loss_function(logits.float(), targets.long())
            results['inner_losses'][step] = inner_loss.item()

            if (step == 0) and is_classification_task:
                results['accuracy_before'] = compute_accuracy(logits, targets)

            model.zero_grad()
            params = update_parameters(model, inner_loss,
                step_size=step_size, params=params,
                first_order=(not model.training) or first_order)

        return params, results


    def train(self, dataloader, max_batches=500, verbose=True, pem_eta=0.9, **kwargs):
        a=[]

        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results,test_inputs in self.train_iter(dataloader, max_batches=max_batches, pem_eta=pem_eta):
                pbar.update(1)
                postfix = {'loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    postfix['accuracy'] = '{0:.4f}'.format(
                        np.mean(results['accuracies_after']))
                pbar.set_postfix(**postfix)
               # a.append[np.fromiter(test_inputs, float)]
               
            #a1 = np.concatenate(a, axis=0)
             
                
        return test_inputs

    def train_iter(self, dataloader, max_batches=500, pem_eta=0.9):
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                self.optimizer.zero_grad()
                batch = tensors_to_device(batch, device=self.device)
                
                if random.random() > pem_eta:
                    batch11= np.array(batch['train'])
                    #print("111111")
                    np.save('D:\\伪监督故障诊断\\数据分析\\batchyuan_train.npy', batch11)
                    batch12= np.array(batch['test'])
                    #print("2211")
                    np.save('D:\\伪监督故障诊断\\数据分析\\batchyuan_test.npy', batch12)                
               
                    nclasses = dataloader.dataset.num_classes_per_task
                    support_size = dataloader.dataset.dataset_transform.splits['train']
                    query_size = dataloader.dataset.dataset_transform.splits['test']
                    batch,a1s,a2s,a3s = self.get_new_samples(batch, support_size, query_size, nclasses)
                    batch22= np.array(batch['train'])
                    #print("2222")
                    np.save('D:\\伪监督故障诊断\\数据分析\\batchlv_train.npy', batch22)
                    batch21= np.array(batch['test'])
                   # print("1122")
                    np.save('D:\\伪监督故障诊断\\数据分析\\batchlv_test.npy', batch21)  
                    np.save('D:\\伪监督故障诊断\\数据分析\\entropy_is.npy', a1s) 
                    np.save('D:\\伪监督故障诊断\\数据分析\\entropys.npy', a2s)  
                    np.save('D:\\伪监督故障诊断\\数据分析\\entropy_probs.npy', a3s)  
            

                outer_loss, results,_ ,_,test_inputs= self.get_outer_loss(batch,Train=False)
                yield results,test_inputs

                outer_loss.backward()
                self.optimizer.step()

                num_batches += 1

    def evaluate(self, dataloader, max_batches=500, verbose=True, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        bb=[]
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results,test_logits,test_targets,b in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                        - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)
                #bb.append[b.numpy()]
               
            #bb1 = np.concatenate(bb, axis=0)

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['accuracies_after'] = mean_accuracy

        return mean_results,test_logits,test_targets,b

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                _, results,test_logits,test_targets,b = self.get_outer_loss(batch,Train=False)
                
                yield results,test_logits,test_targets,b

                num_batches += 1

    def get_samples_from_k_cluster(self, params, class_ind, cluster_ind, query_size, nclasses, k=5, keep_rate=0.75):
        ind_cand = self.ind_sorted[cluster_ind][1:k+1]
        ## select cluster with entropy
        logits_list = []
        entropy_max = -10000
        i1=[]
        entropy1=[]
        prob1=[]
        for count, i in enumerate(ind_cand):
            x = self.c_list[i]
            x = torch.tensor(x)
            with torch.set_grad_enabled(False):
                test_logit = self.eval_model(x, params=params)
            unlabel_size = x.shape[0]
            # caculate entropy
            _, t_label = torch.max(test_logit, dim=1)
            ### select the cluster with max entropy
            hist = torch.histc(t_label.float(), bins=nclasses, max=nclasses - 1)
            prob = hist * 1. / unlabel_size
            entropy = -sum(prob * torch.log(prob + 1e-8))
            logits_list.append(test_logit)
            #print('cluster ind:{}, entropy:{}, prob:{}'.format(i, entropy, prob))
            i1.append(i)
            entropy1.append(entropy)
            prob1.append(prob)
            

            if entropy > entropy_max:
                best_ind = count
                entropy_max = entropy
        
        iii= np.array(i1)
           
        np.save('D:\\伪监督故障诊断\\数据分析\\entropy_i.npy', iii)
        iii1= np.array(entropy1)         
        np.save('D:\\伪监督故障诊断\\数据分析\\entropy.npy', iii1)           
        iii2= np.array(prob1)    
        np.save('D:\\伪监督故障诊断\\数据分析\\entropy_prob.npy', iii2)  
        i = ind_cand[best_ind]
        test_logit = logits_list[best_ind]
        ## filter out noisy samples
        unlabel_size = test_logit.shape[0]
        n_keep = max(math.ceil(unlabel_size * keep_rate), query_size)

        # sort the logits in current cluster and filter out noisy samples
        score = test_logit[:, class_ind]
        sorted_score, ind = torch.sort(score, descending=True)
        ind = ind[:n_keep]
        ind0 = select_index(n_keep, query_size)
        ind = ind[ind0]
        x = self.c_list[i][ind.cpu()]
        np.save('D:\\伪监督故障诊断\\数据分析\\sample_lv.npy', x)  
        xx=self.c_list[i]
        np.save('D:\\伪监督故障诊断\\数据分析\\sample.npy', xx)  
        
        return x,iii,iii1,iii2

    def get_new_samples(self, batch, support_size, query_size, nclasses):
        train_inputs, train_targets, train_cluster_labels = batch['train'][0], batch['train'][1], batch['train'][2]
        test_inputs = batch['test'][0]
        B, N, C, H = train_inputs.shape
        for i in range(B):
            inputs, targets, labels = train_inputs[i], train_targets[i], train_cluster_labels[i]
            params, adaptation_results = self.adapt(inputs, targets, self.eval_model,
                                                    is_classification_task=True,
                                                    num_adaptation_steps=self.num_adaptation_steps,
                                                    step_size=self.step_size, first_order=self.first_order)
            tragets0, labels0 = targets.view(nclasses, support_size)[:,0], labels.view(nclasses, support_size)[:, 0]
            count = 0
            for t, l in zip(tragets0, labels0):
                query_samples0,a1,a2,a3 = self.get_samples_from_k_cluster(params, t, l, query_size, nclasses) # NxCxHxW
                query_samples0 = torch.tensor(query_samples0)
                if count==0:
                    query_samples=query_samples0
                    a1s=a1
                    a2s=a2
                    a3s=a3
                else:
                    query_samples = torch.cat([query_samples, query_samples0], dim=0)
                    a1s=np.concatenate((a1s,a1),axis=0)
                    a2s=np.concatenate((a2s,a2),axis=0)
                    a3s=np.concatenate((a3s,a3),axis=0)
                    
                count=count+1
            query_samples=tensors_to_device(query_samples, device=self.device)
            test_inputs[i] = query_samples
        batch['test'][0] = test_inputs

        return batch,a1s,a2s,a3s

MAML = ModelAgnosticMetaLearning