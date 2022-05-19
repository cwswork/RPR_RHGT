import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

from align import pre_loadKGs
from autil import fileUtil
from align.model_htrans import Align_Htrans


class align_set100K():
    def __init__(self, myconfig):
        super(align_set100K, self).__init__()
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
            self.align_model = self.align_model.cuda()
        ### optimizer
        if myconfig.optim_type == 'Adagrad':
            self.optimizer = torch.optim.Adam(self.align_model.model_params, lr=myconfig.learning_rate,
                            weight_decay=myconfig.weight_decay)  # weight_decay =5e-4
        else:
            self.optimizer = torch.optim.SGD(self.align_model.model_params, lr=myconfig.learning_rate,
                            weight_decay=myconfig.weight_decay)

    ## model train
    def model_run(self, beg_epochs=0):
        t_begin = time.time()

        self.bad_counter, self.min_validloss_counter = 0, 0
        self.best_hits1, self.best_epochs = 0, -1  # best test
        self.min_valid_loss = sys.maxsize
        for epochs_i in range(beg_epochs, self.myconfig.train_epochs):
            ## Train
            train_hits1 = self.runTrain(epochs_i)

            ## Test
            if epochs_i >= self.myconfig.start_valid and epochs_i % self.myconfig.eval_freq == 0:
                break_re = self.runTest(epochs_i)
                if self.myconfig.early_stop and break_re:
                    break

        # printing
        self.save_model(epochs_i, 'last')  # last_epochs
        self.myprint("Optimization Finished!")
        self.myprint('Last epoch-{:04d}:'.format(epochs_i))
        self.myprint('Best Test epoch-{:04d}:'.format(self.best_epochs))

        # Last Test
        self.runTest(epochs_i=epochs_i, is_save=True)  # Testing

        # Best Test
        self.myprint('Best epoch-{:04d}:'.format(self.best_epochs))
        if epochs_i != self.best_epochs:
            best_savefile = '{}-epochs-{}-{}.pkl'.format(self.best_mode_pkl_title, self.best_epochs, 'best')
            self.load_model(best_savefile, self.best_epochs)
        self.runTest(epochs_i=self.best_epochs, is_save=True)  # Testing

        self.myprint("Total time elapsed: {:.4f}s".format(time.time() - t_begin))


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
        torch.cuda.empty_cache()
        train_loss.backward()
        loss_float = train_loss.data.item()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        self.optimizer.step()
        self.myprint('Epoch-{:04d}: Train_loss-{:.8f}, cost time-{:.4f}s'.format(
            epochs_i, loss_float, time.time() - t_epoch))

        # Accuracy
        if epochs_i % 5 == 0:
            [hits1_L, result_str_L, Hits_list_L], _ = self.align_model.accuracy(link_type=1)
            self.myprint('Train==' + result_str_L + ';skip_w:'+ skip_w)
        else:
            hits1_L = 0

        return hits1_L

    def runTest(self, epochs_i, is_save=False):
        break_re = False
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
        testRe_print = '++TEST Result++Epochs-{:04d}: \n{}\n{}'.format(epochs_i, result_str_L, result_str_R)
        self.myprint(testRe_print)

        # ********************no early stop********************************************
        if hits1_L >= self.best_hits1:
            self.best_hits1 = hits1_L
            self.best_epochs = epochs_i
            self.bad_counter = 0
            self.save_model(epochs_i, 'best')  # save model
            self.myprint('==Test==Epoch-{:04d}, better result, best_hits1:{:.4f}..'.format(epochs_i, self.best_hits1))
            ###
            with open(self.best_mode_pkl_title + '_Result.txt', "a") as ff:
                ff.write(testRe_print)
        else:
            self.bad_counter += 1
            self.myprint('==bad_counter++:' + str(self.bad_counter))
            # bad model, stop train
            if self.bad_counter == self.myconfig.patience:  # patience=20
                self.myprint('==bad_counter, stop training.')
                break_re = True

        if is_save:  # only reTest
            self.myprint('++++++++ Save TEST Result ++++++++')
            save_file = '{}_{}'.format(self.best_mode_pkl_title, epochs_i)
            fileUtil.save_list2txt(save_file + '_Left_hitslist.txt', Hits_list_L)
            fileUtil.save_list2txt(save_file + '_Right_hitslist.txt', Hits_list_R)

        return break_re


    def save_model(self, better_epochs_i, epochs_name):  # best-epochs
        # save model to file
        model_savefile = '{}-epochs-{}-{}.pkl'.format(self.best_mode_pkl_title, better_epochs_i, epochs_name)
        model_state = dict()
        model_state['align_layer'] = self.align_model.state_dict()
        torch.save(model_state, model_savefile)


    def load_model(self, model_savefile, epoch_i):  # best-epochs
        # load model to file
        self.myprint('\nLoading file: {} - {}th epoch'.format(model_savefile, epoch_i))
        if self.myconfig.is_cuda:
            checkpoint = torch.load(model_savefile)
        else:
            checkpoint = torch.load(model_savefile, map_location='cpu')  # GPU->CPU
        self.align_model.load_state_dict(checkpoint['align_layer'])

