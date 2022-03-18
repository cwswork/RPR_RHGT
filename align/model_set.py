import os
import sys
import time
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn

from align import pre_loadKGs
from autil import fileUtil
from align.model_htrans import Align_Htrans


class align_set():
    def __init__(self, myconfig):
        super(align_set, self).__init__()
        self.myconfig = myconfig
        self.myprint = myconfig.myprint
        self.best_mode_pkl_title = myconfig.output + myconfig.time_str

        # Load KGs data
        kgsdata_file = myconfig.out_temp + 'kgsdata.pkl'
        if os.path.exists(kgsdata_file):
            kgs_data = fileUtil.loadpickle(kgsdata_file)
        else:
            kgs_data = pre_loadKGs.load_KGs_data(myconfig)
            fileUtil.savepickle(kgsdata_file, kgs_data)
        self.myprint('====Finish Load kgs_data====' + kgsdata_file)

        ## Model and optimizer ######################
        self.align_model = Align_Htrans(kgs_data, myconfig)
        if myconfig.is_cuda:
            self.align_model = self.align_model.cuda(myconfig.device)
        ### optimizer
        if myconfig.optim_type == 'Adagrad':
            self.optimizer = torch.optim.Adam(self.align_model.model_params, lr=myconfig.learning_rate,
                            weight_decay=myconfig.weight_decay)  # 权重衰减（参数L2损失）weight_decay =5e-4
        else:
            self.optimizer = torch.optim.SGD(self.align_model.model_params, lr=myconfig.learning_rate,
                            weight_decay=myconfig.weight_decay)

    ## model train
    def model_run(self, beg_epochs=0):
        t_begin = time.time()

        self.bad_counter, self.min_validloss_counter = 0, 0
        self.best_hits1, self.best_epochs = 0, -1  # best Valid
        self.best_test_hits1, self.best_test_epochs, = 0, -1  # best Test # Test Del
        self.min_valid_loss = sys.maxsize
        for epochs_i in range(beg_epochs, self.myconfig.train_epochs):
            ## Train
            train_hits1 = self.runTrain(epochs_i)

            ## Valid
            if epochs_i >= self.myconfig.start_valid and epochs_i % self.myconfig.eval_freq == 0:
                break_re, valid_hits1 = self.runValid(epochs_i)
                if self.myconfig.early_stop and break_re:
                    break

        self.save_model(epochs_i, 'last')  # last_epochs
        self.myprint("Optimization Finished!")
        self.myprint('Last epoch-{:04d}:'.format(epochs_i))
        self.myprint('Best epoch-{:04d}:'.format(self.best_epochs))
        self.myprint('Best Test epoch-{:04d}:'.format(self.best_test_epochs)) # Test Del

        # Last Test
        self.runTest(epochs_i=epochs_i, is_save=True)  # Testing

        # Best Test, load model
        self.myprint('Best epoch-{:04d}:'.format(self.best_epochs))
        if epochs_i != self.best_epochs:
            best_savefile = '{}-epochs-{}-{}.pkl'.format(self.best_mode_pkl_title, self.best_epochs, 'best')
            self.myprint('\nLoading file: {} - {}th epoch'.format(best_savefile, self.best_epochs))
            self.reRunTest(best_savefile, self.best_epochs)
        self.myprint("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        self.myprint('model arguments:' + self.myconfig.get_param())

    def runTrain(self, epochs_i):
        t_epoch = time.time()
        # Model trainning
        # Forward pass
        self.align_model.train()
        self.optimizer.zero_grad()

        # model action  4 loss
        skip_w = self.align_model.forward()
        train_loss = self.align_model.get_loss(epochs_i, link_type=1)
        # Backward and optimize
        train_loss.backward()
        self.optimizer.step()
        self.myprint('Epoch-{:04d}: Train_loss-{:.8f}, cost time-{:.4f}s'.format(
            epochs_i, train_loss.data.item(), time.time() - t_epoch))

        # Accuracy
        if epochs_i % 5 == 0:
            [hits1_L, result_str_L, Hits_list_L], _ = self.align_model.accuracy(link_type=1)
            self.myprint('Train==' + result_str_L + ';skip_w:'+ skip_w)
        else:
            hits1_L = 0
        return hits1_L

    def runValid(self, epochs_i):
        t_epoch = time.time()
        break_re = False
        with torch.no_grad():
            # Forward pass
            self.align_model.eval()
            # Model trainning
            self.align_model.forward()
            # loss
            valid_loss = self.align_model.get_loss(epochs_i, link_type=2)
            loss_float = valid_loss.data.item()
            [hits1_L, result_str_L, Hits_list_L], _ = self.align_model.accuracy(link_type=2)

            #if loss_alpha == None:
            self.myprint('Epoch-{:04d}: Valid_loss-{:.8f}, cost time-{:.4f}s'.format(
                epochs_i, valid_loss.data.item(), time.time() - t_epoch))
            self.myprint('==Valid==' + result_str_L)

        # ********************no early stop********************************************
        # save best model in valid
        if hits1_L >= self.best_hits1:
            self.best_hits1 = hits1_L
            self.best_epochs = epochs_i
            self.bad_counter = 0
            self.save_model(epochs_i, 'best')
            self.myprint('==Valid==Epoch-{:04d}, better result, best_hits1:{:.4f}..'.format(epochs_i, self.best_hits1))
        else:
            # no best, but save model every 10 epochs
            if epochs_i % self.myconfig.eval_save_freq == 0:
                self.save_model(epochs_i, 'eval')

            self.bad_counter += 1
            self.myprint('==bad_counter++:' + str(self.bad_counter))
            # bad model, stop train
            if self.bad_counter == self.myconfig.patience:
                self.myprint('==bad_counter, stop training.')
                break_re = True

        # Verification set loss continuous decline also stop training!
        if loss_float <= self.min_valid_loss:
            self.min_valid_loss = loss_float
            self.min_validloss_counter = 0
            self.myprint('Epoch-{:04d}, min_valid_loss:{:.8f}..'.format(epochs_i, self.min_valid_loss))
        else:
            self.min_validloss_counter += 1
            self.myprint('==min_validloss_counter++:{}'.format(self.min_validloss_counter))
            if self.min_validloss_counter == self.myconfig.patience_minloss:
                self.myprint('==bad min_valid_loss, stop training.')
                break_re = True

        return break_re, hits1_L


    def runTest(self, epochs_i, is_save=False):

        with torch.no_grad():
            # 2 Forward pass
            self.align_model.eval()
            # 3 model action
            self.align_model.forward()

            # 4 Accuracy
            Left_re, Right_re = self.align_model.accuracy(link_type=3)
            [hits1_L, result_str_L, Hits_list_L] = Left_re
            result_str_L = "==From left:" + result_str_L
            [hits1_R, result_str_R, Hits_list_R] = Right_re
            result_str_R = "==From right:" + result_str_R

            # From left
            # self.myprint('++++++++TEST Result++++++++')
            testRe_print = '++TEST Result++Epochs-{:04d}: \n{}\n{}'.format(epochs_i, result_str_L, result_str_R)
            self.myprint(testRe_print)

        ### Test Del
        if hits1_L >= self.best_test_hits1:
            self.best_test_epochs = epochs_i
            self.best_test_hits1 = hits1_L
            with open(self.best_mode_pkl_title + '_Result.txt', "a") as ff:
                ff.write(testRe_print)

        ###############
        if is_save:  # only reTest
            self.myprint('++++++++ Save TEST Result ++++++++')
            save_file = '{}_{}'.format(self.best_mode_pkl_title, epochs_i)
            fileUtil.save_list2txt(save_file + '_Left_hitslist.txt', Hits_list_L)
            fileUtil.save_list2txt(save_file + '_Right_hitslist.txt', Hits_list_R)


    def save_model(self, better_epochs_i, epochs_name):  # best-epochs
        model_savefile = '{}-epochs-{}-{}.pkl'.format(self.best_mode_pkl_title, better_epochs_i, epochs_name)
        model_state = dict()
        model_state['align_layer'] = self.align_model.state_dict()
        model_state['myconfig'] = self.myconfig
        torch.save(model_state, model_savefile)


    def reRunTrain(self, model_savefile, beg_epochs, is_cuda=False):  # best-epochs
        # load model to file
        self.myprint('\nLoading file: {} - {}th epoch'.format(model_savefile, beg_epochs))
        if is_cuda:
            checkpoint = torch.load(model_savefile)
        else:
            checkpoint = torch.load(model_savefile, map_location='cpu')  # GPU->CPU
        self.align_model.load_state_dict(checkpoint['align_layer'])
        self.myconfig = checkpoint['myconfig']
        self.model_run(beg_epochs=beg_epochs)


    def reRunTest(self, model_savefile, epoch_i):  # best-epochs
        if self.myconfig.is_cuda:
            checkpoint = torch.load(model_savefile)
        else:
            checkpoint = torch.load(model_savefile, map_location='cpu')  # GPU->CPU
        self.align_model.load_state_dict(checkpoint['align_layer'])
        self.runTest(epochs_i=epoch_i, is_save=True)  # Testing
