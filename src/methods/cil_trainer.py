from typing import Tuple
from argparse import Namespace
from loguru._logger import Logger

import torch.nn.functional as F
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from .base import Trainer
from utils import *
from dataloader.data_utils import *
from cnn.features import create_feature_extractor, create_film_adapter
from cnn.efficientnet import efficientnet

class CIL_Trainer(Trainer):
    def __init__(self, args: Namespace, lggr: Logger):
        super().__init__(args)
        self.args = args
        self.args = set_up_datasets(self.args)
        self.lggr = lggr

        if self.args.use_film:
            self.feature_extractor = create_feature_extractor().to(self.args.device)
            self.feature_adaptation_network = create_film_adapter(self.feature_extractor).to(self.args.device)
        else:
            self.feature_extractor = efficientnet().to(self.args.device)
        self.feature_extractor.eval()        
        
        self.linear_head = nn.Linear(self.feature_extractor.output_size, args.base_class).to(self.args.device)
        self.drop_out = nn.Dropout(p=0.5)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def get_optimizer_scheduler(self):
        model = self.feature_adaptation_network if self.args.use_film else self.feature_extractor
        optimizer = torch.optim.Adam(list(model.parameters()) + list(self.linear_head.parameters()), lr=self.args.base_lr)
        if self.args.few_shots:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 100], gamma=0.3)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.7) #or gamma=0.3

        return optimizer, scheduler

    def get_dataloader(self, images, labels, use_test_transf=False, shuffle=True):
        if self.args.few_shots:
            tr_transf, tst_transf = None, None
        else:
            tr_transf, tst_transf = get_tranforms(self.args)
        transf = tst_transf if use_test_transf else tr_transf
        bsize = self.args.test_batch_size if use_test_transf else self.args.batch_size

        flag = self.args.dataset == 'letters' and not self.args.few_shots
        dataset = Dataset_transform_V2(images, labels) if flag else Dataset_transform(images, labels, transform=transf)
        dtloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=bsize, shuffle=shuffle, num_workers=self.args.num_workers, pin_memory=True)

        return dtloader

    def get_final_loader(self, cl_loader=None):
        if cl_loader is not None:
            trainloader = self.get_dataloader(cl_loader.train_images, cl_loader.train_labels.numpy(), shuffle=False)
            testloader = self.get_dataloader(cl_loader.test_images, cl_loader.test_labels.numpy(), shuffle=False)
        else:
            cl_loader = get_cl_dataloader(self.args)
            imgs_all, labels_all = [], []
            for session, session_data in enumerate(cl_loader):
                train_x, train_y, test_x, test_y = session_data
                if isinstance(train_x, list):
                    imgs_all += train_x
                else:
                    imgs_all.append(train_x)
                labels_all.append(train_y)
                if session == self.args.sessions - 1:
                    testloader = self.get_dataloader(test_x, test_y, use_test_transf=True, shuffle=False)

            if isinstance(train_x, list):
                trainloader = self.get_dataloader(imgs_all, np.concatenate(labels_all), use_test_transf=True, shuffle=False)
            else:
                trainloader = self.get_dataloader(np.vstack(imgs_all), np.concatenate(labels_all), use_test_transf=True, shuffle=False)
        return trainloader, testloader

    def get_train_test_loader(self, cl_loader=None):
        #"""
        if cl_loader is not None:
            train_x, train_y, test_x, test_y = next(iter(cl_loader))
            cl_loader.reset_session()
        else:
            train_x, train_y, test_x, test_y = next(iter(get_cl_dataloader(self.args)))
        trainloader = self.get_dataloader(train_x, train_y, use_test_transf=False, shuffle=True)
        testloader = self.get_dataloader(test_x, test_y, use_test_transf=True, shuffle=False)
        
        return trainloader, testloader
    
    def train_eval_org(self):
        cl_loader = get_cl_dataloader(self.args)
        if self.args.few_shots:
            trainloader, testloader = self.get_train_test_loader(cl_loader)         
        else: 
            trainloader, testloader = self.get_train_test_loader()         

        total_train_time = self.train_learner(trainloader)
        cl_loader = get_cl_dataloader(self.args)
        test_acc_session = np.zeros(self.args.sessions)
        imgs_all, labels_all = [], []
        for session, session_data in enumerate(cl_loader):
            train_x, train_y, test_x, test_y = session_data
            if isinstance(train_x, list):
                self.lggr.info("\n\n*** Session {} ***\ntrain_x: {}  test_x: {}" .format(session, len(train_x), len(test_x)))
                imgs_all += train_x
            else:
                self.lggr.info("\n\n*** Session {} ***\ntrain_x: {}  test_x: {}" .format(session, train_x.shape, test_x.shape))
                imgs_all.append(train_x)
            labels_all.append(train_y)
            
            if isinstance(train_x, list):
                trainloader_cl = self.get_dataloader(imgs_all, np.concatenate(labels_all), use_test_transf=True, shuffle=False)
            else:
                trainloader_cl = self.get_dataloader(np.vstack(imgs_all), np.concatenate(labels_all), use_test_transf=True, shuffle=False)
                
            testloader_cl = self.get_dataloader(test_x, test_y, use_test_transf=True, shuffle=False)
            assert len(np.unique(np.concatenate(labels_all))) == len(np.unique(test_y))
            self.lggr.info("Unique classes: {}" .format(np.unique(np.concatenate(labels_all))))
           
            _, tsa_cl = self.test_cl(trainloader_cl, testloader_cl)
            test_acc_session[session] = 100 * tsa_cl
        
        self.lggr.info('Total Training time  %.2f mins' %(total_train_time / 60))
        self.lggr.info("\n" + 50 * "" + "*** Accuracy per session ***")
        self.lggr.info(test_acc_session.round(2))

    def train_eval_fscil(self):
        trainloader, testloader = get_downloader_session(self.args, 0)
        total_train_time = self.train_learner(trainloader)
        test_acc_session = np.zeros(self.args.sessions)
        
        self.lggr.info("\n\n*** Session {} ***\ntrain_x: {}  test_x: {}" .format(0, len(trainloader.dataset), len(testloader.dataset)))
        _, tsa_cl = self.test_cl(trainloader, testloader)
        test_acc_session[0] = 100 * tsa_cl
        for session in range(1, self.args.sessions):
            trainloader, testloader = get_downloader_session(self.args, session)
            self.lggr.info("\n\n*** Session {} ***\ntrain_x: {}  test_x: {}" .format(session, len(trainloader.dataset), len(testloader.dataset)))
            _, tsa_cl = self.test_cl(trainloader, testloader)
            test_acc_session[session] = 100 * tsa_cl
            
        self.lggr.info('Total Training time  %.2f mins' %(total_train_time / 60))
        self.lggr.info("\n" + 50 * "" + "*** Accuracy per session ***")
        self.lggr.info(test_acc_session.round(2))
        
    def train_eval(self):
        set_seed(self.args.seed)
        if self.args.fscil:
            self.train_eval_fscil()
        else:
            self.train_eval_org()
        

    def train_learner(self, trainloader) -> float:
        self.lggr.info("\n" + 50 * "" + "*** Fine-tuning at Session 0 ***")
        optimizer, scheduler = self.get_optimizer_scheduler()
        total_train_time = 0.
        for epoch in range(self.args.epochs_base):
            start_time = time.time()
            # train base sess
            tl, ta = self.base_train(trainloader, optimizer)
            train_time = time.time() - start_time
            total_train_time += train_time
            
            self.lggr.info('Epoch %d/%d, (train) loss=%.4f, acc=%.2f%%, time=%.1f seconds' %(epoch + 1, self.args.epochs_base, tl, 100*ta, time.time() - start_time))
            scheduler.step()
            
        ensure_path(os.path.join(self.args.dir_save, "saved_model"))
        save_model_dir = os.path.join(self.args.dir_save, "saved_model", 'fine_tuned_model_at_Session_0.pt')
        if self.args.use_film:
            with torch.inference_mode():
                self.film_params = self.feature_adaptation_network(None)
            torch.save(dict(params=self.feature_adaptation_network.state_dict()), save_model_dir)
        else:
            torch.save(dict(params=self.feature_extractor.state_dict()), save_model_dir)
            
        return total_train_time      

    def base_train(self, trainloader, optimizer: Optimizer) -> Tuple[float, float]:
        tl = Averager()
        ta = Averager()
        torch.set_grad_enabled(True)
        if self.args.use_film:
            film_params = self.feature_adaptation_network(None)
            self.feature_extractor.eval()
        else:
            self.feature_extractor.train()
            
        for imgs_b, labels_b in trainloader:
            labels_b = labels_b.type(torch.long)
            imgs_b, labels_b = imgs_b.cuda(), labels_b.cuda()
            
            if self.args.use_film:
                feat_b = self.feature_extractor(imgs_b, film_params)
            else:
                feat_b = self.feature_extractor(imgs_b)
                
            logits = self.linear_head(self.drop_out(feat_b))
            loss = self.criterion(logits, labels_b)
            
            acc = count_acc(logits, labels_b)
            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        tl = tl.item()
        ta = ta.item()
                
        return tl, ta

    def test(self, testloader, epoch=None) -> Tuple[float, float]:
        vl, va = Averager(), Averager()
        self.feature_extractor.eval()
        with torch.inference_mode():
            for imgs_b, labels_b in testloader:
                labels_b = labels_b.type(torch.long)
                imgs_b, labels_b = imgs_b.cuda(), labels_b.cuda()
                
                logits = self.linear_head(self.feature_extractor(imgs_b, self.film_params))
                loss = self.criterion(logits, labels_b)
                acc = count_acc(logits, labels_b)
                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()
            if epoch is None:
                self.lggr.info('test loss={:.4f} | test acc={:.42}%'.format(vl, 100 * va))
            else:
                self.lggr.info('\nEpoch ({}) | test loss={:.4f} | test acc={:.2f}%'.format(epoch, vl, 100 * va))        

        return vl, va

    def compute_feat_by_batch(self, dt_loader) -> Tuple[Tensor, Tensor]:
        features = []
        labels = []
        for imgs_b, labels_b in dt_loader:
            labels_b = labels_b.type(torch.long)
            imgs_b, labels_b = imgs_b.cuda(), labels_b.cuda()
            if self.args.use_film:
                feat_batch = self.feature_extractor(imgs_b, self.film_params)
            else:
                feat_batch = self.feature_extractor(imgs_b)
            features.append(feat_batch)
            labels.append(labels_b)

        return torch.vstack(features), torch.concat(labels)

    def test_cl(self, tr_loader, tst_loader) -> Tuple[float, float]:
        with torch.inference_mode():
            feat_train, labels_train = self.compute_feat_by_batch(tr_loader)
            coeff, bias = compute_lda_head(feat_train, labels_train, self.args)
            vl, va = self.test_acc(tst_loader, coeff, bias)

        return vl, va

    def test_acc(self, dt_loader, coeff, bias) -> Tuple[float, float]:
        vl, va = Averager(), Averager()
        for imgs_b, labels_b in dt_loader:
            labels_b = labels_b.type(torch.long)
            imgs_b, labels_b = imgs_b.cuda(), labels_b.cuda()
            
            feat_b = self.feature_extractor(imgs_b, self.film_params)
            logits = torch.mm(feat_b, coeff) - 0.5 * bias
            loss = self.criterion(logits, labels_b)
            acc = count_acc(logits, labels_b)
            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()

        self.lggr.info('CL-all test, loss={:.4f} acc={:.2f}%'.format(vl, 100 * va))

        return vl, va
