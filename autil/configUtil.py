from __future__ import division
from __future__ import print_function
import os
import random
import re
import time

import json
import numpy as np
import torch

class configClass():
    def __init__(self, args_file, datasets, division='1/'):
        args = ARGs(args_file)
        self.datasetPath = args.datasetPath + datasets + 'pre/'
        self.tt_path = args.datasetPath + datasets + '721_5fold/' + division
        self.output = args.output + datasets + division

        # embed file
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        self.out_temp = self.datasetPath + 'temp/' + division
        if not os.path.exists(self.out_temp):
            os.makedirs(self.out_temp)

        self.seed = args.seed
        self.time_str = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))

        #
        self.e_dim = args.e_dim
        self.embed_type = args.embed_type
        self.dropout = 0
        #self.attn_heads = 1

        #
        self.optim_type = args.optim_type
        self.patience = args.patience
        self.patience_minloss = args.patience_minloss
        self.metric = args.metric
        self.train_epochs = args.train_epochs
        self.is_cuda = True

        #
        self.early_stop = args.early_stop
        self.start_valid = args.start_valid
        self.eval_freq = args.eval_freq
        self.eval_save_freq = args.eval_save_freq  #  20

        #
        self.top_k = args.top_k
        self.neg_k = args.neg_k  # number of negative samples for each positive one

        #
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.LeakyReLU_alpha = args.LeakyReLU_alpha
        self.dropout = args.dropout
        self.embed_type = 1  # 2vectorList.json
        #self.learning_rate = 0.001
        self.dropout = args.dropout
        self.gamma_rel = args.gamma_rel
        #self.beta1 = args.beta1
        self.l_beta = args.l_beta
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads

    def get_param(self):
        self.model_param = 'eps_' + str(self.train_epochs) + \
            '-nk_' + str(self.neg_k) + \
            '-me_' + str(self.metric) + \
            '-lr_' + str(self.learning_rate) + \
            '-ed_' + str(self.e_dim) + \
            '-hn_' + str(self.n_heads) + \
            '-ly_' + str(self.n_layers) + \
            '-et_' + str(self.embed_type) + \
            '-gr_' + str(self.gamma_rel) + \
            '-lbe_' + str(self.l_beta) + \
            '-pa_' + str(self.patience) + \
            '-drop_' + str(self.dropout)

        return self.model_param

    def set_myprint(self, runfile, issave=True):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if issave:
            print_Class = Myprint(self.output, 'train_log' + self.time_str + '.txt')
            if not os.path.exists(self.output):
                print('output not exists' + self.output)
                os.makedirs(self.output)
            self.myprint = print_Class.print
        else:
            self.myprint = print

        self.myprint("start==" + self.time_str + ": " + runfile)
        self.myprint('output Path:' + self.output)
        self.myprint('cuda.is_available:' + str(self.is_cuda))
        self.myprint('model arguments:' + self.get_param())


#####################################
class Myprint:
    def __init__(self, filePath, filename):
        if not os.path.exists(filePath):
            print('output not exists' + filePath)
            os.makedirs(filePath)

        self.outfile = filePath + filename

    def print(self, print_str):
        print(print_str)
        '''保存log文件'''
        with open(self.outfile, 'a', encoding='utf-8') as fw:
            fw.write('{}\n'.format(print_str))

#############################
class ARGs:
    ''' 加载配置问卷 args/** .json '''
    def __init__(self, file_path):
        args_dict = loadmyJson(file_path)
        for k, v in args_dict.items():
            setattr(self, k, v)

################################################
# Load JSON File
def loadmyJson(JsonPath):
    try:
        srcJson = open(JsonPath, 'r', encoding= 'utf-8')
    except:
        print('cannot open ' + JsonPath)
        quit()

    dstJsonStr = ''
    for line in srcJson.readlines():
        if not re.match(r'\s*//', line) and not re.match(r'\s*\n', line):
            dstJsonStr += cleanNote(line)

    # print dstJsonStr
    dstJson = {}
    try:
        dstJson = json.loads(dstJsonStr)
    except:
        print(JsonPath + ' is not a valid json file')

    return dstJson


def cleanNote(line_str):
    qtCnt = cmtPos = 0
    rearLine = line_str
    while rearLine.find('//') >= 0:
        slashPos = rearLine.find('//')
        cmtPos += slashPos
        headLine = rearLine[:slashPos]
        while headLine.find('"') >= 0:
            qtPos = headLine.find('"')
            if not isEscapeOpr(headLine[:qtPos]):
                qtCnt += 1
            headLine = headLine[qtPos+1:]
            # print qtCnt
        if qtCnt % 2 == 0:
            # print self.instr[:cmtPos]
            return line_str[:cmtPos]
        rearLine = rearLine[slashPos+2:]
        # print rearLine
        cmtPos += 2

    return line_str


def isEscapeOpr(instr):
    if len(instr) <= 0:
        return False
    cnt = 0
    while instr[-1] == '\\':
        cnt += 1
        instr = instr[:-1]
    if cnt % 2 == 1:
        return True
    else:
        return False
