import os
import time
import torch

from align100K import model_set100K
from autil.configUtil import configClass


def setRunmodel(dataset, division, model_type, outname, l_beta=0.3, gamma_rel=10, is_cuda=False, device='cpu'):
    print('\n---------------------------------------')
    myconfig = configClass('../args/args_15K.json', datasets=dataset, division=division)

    myconfig.is_cuda = is_cuda and torch.cuda.is_available()  # cuda
    device = torch.device("cuda:" + device if myconfig.is_cuda else "cpu")
    print("GPU No: {}".format(device))
    if myconfig.is_cuda:
        myconfig.device = device
    ###### Model
    if 'path' in model_type:
        myconfig.patience = 20
    else:
        myconfig.patience = 20
    myconfig.n_layers = 1  # 4
    myconfig.n_heads = 4  # 4
    myconfig.e_dim = 200 #300 ->100

    myconfig.l_beta = l_beta  #
    myconfig.gamma_rel = gamma_rel
    myconfig.model_type = model_type  # rel rel_path
    ##################
    outname = '({}{})'.format(myconfig.model_type, outname)
    myconfig.output += 'htrans_' + outname + '/' + myconfig.time_str + '/'
    myconfig.set_myprint(os.path.realpath(__file__))
    myconfig.myprint(
        "\n==train align_model, e_dim:{}, gamma_rel:{}, l_beta:{}".format(myconfig.e_dim, myconfig.gamma_rel,
                                                                          myconfig.l_beta))
    mymodel = model_set100K.align_set100K(myconfig)
    mymodel.model_run()
    mymodel.myprint("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    print('---------------------------------------------')

if __name__ == '__main__':
    #datasets = 'WN31/EN_DE_15K_V1/' # EN_DE_15K_V1、EN_FR_15K_V1
    #datasets = 'DWY100K/dbp_yg/' # DWY100K/dbp_wd, DWY100K/dbp_yg
    setRunmodel(dataset='DWY100K/dbp_wd/', division='30/', outname='196_1215', model_type='rel_path', l_beta=0.2)
