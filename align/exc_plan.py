import os
import time
import torch

from align.model_set import align_set
from autil.configUtil import configClass


def setRunmodel(datasets, division, model_type, outname, l_beta=0.3, gamma_rel=10, n_layers=1, is_cuda=False, device='cpu'):
    print('\n---------------------------------------')
    myconfig = configClass('../args/args_15K.json', datasets=datasets, division=division)

    # Run setting
    myconfig.is_cuda = is_cuda and torch.cuda.is_available()  # cuda
    device = torch.device("cuda:" + device if myconfig.is_cuda else "cpu")
    print("GPU No.: {}".format(device))
    if myconfig.is_cuda:
        myconfig.device = device
    ######
    if 'path' in model_type:
        myconfig.patience = 40
    else:
        myconfig.patience = 30
    myconfig.n_layers = n_layers  # 4
    myconfig.n_heads = 4  # 4
    myconfig.e_dim = 300

    myconfig.l_beta = l_beta  #
    myconfig.gamma_rel = gamma_rel
    myconfig.model_type = model_type  # rel rel_path
    ##################
    outname = '({}{})'.format(myconfig.model_type, outname)
    myconfig.output += 'htrans' + outname + '/' + myconfig.time_str + '/' # 'htrans' +
    myconfig.set_myprint(os.path.realpath(__file__))
    myconfig.myprint("\n==train align_model, n_layers:{}, n_heads:{}, e_dim:{}, gamma_rel:{}, l_beta:{}".format(myconfig.n_layers,
                                                                                                   myconfig.n_heads,
                                                                                                   myconfig.e_dim,
                                                                                                   myconfig.gamma_rel,
                                                                                                   myconfig.l_beta))
    mymodel = align_set(myconfig)
    mymodel.model_run()
    mymodel.myprint("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    print('---------------------------------------------')


def runAll_division(datasets, outname, model_type='ref', n_layers=1, is_cuda=False, device='cpu'):
    setRunmodel(datasets=datasets, division='1/', n_layers=n_layers, outname=outname, model_type=model_type, is_cuda=is_cuda, device=device)
    setRunmodel(datasets=datasets, division='2/', n_layers=n_layers, outname=outname, model_type=model_type, is_cuda=is_cuda, device=device)
    setRunmodel(datasets=datasets, division='3/', n_layers=n_layers, outname=outname, model_type=model_type, is_cuda=is_cuda, device=device)
    setRunmodel(datasets=datasets, division='4/', n_layers=n_layers, outname=outname, model_type=model_type, is_cuda=is_cuda, device=device)
    setRunmodel(datasets=datasets, division='5/', n_layers=n_layers, outname=outname, model_type=model_type, is_cuda=is_cuda, device=device)


def runRel(outname, n_layers=1, is_cuda=False, device='cpu'):
    # runAll_division(datasets='DBP15K/fr_en(dbp15)/', model_type='rel', is_cuda=is_cuda, device=device)
    runAll_division(datasets='WN31/EN_DE_15K_V2/', model_type='rel', outname=outname, n_layers=n_layers, is_cuda=is_cuda, device=device)
    runAll_division(datasets='WN31/EN_DE_15K_V1/', model_type='rel', outname=outname, n_layers=n_layers, is_cuda=is_cuda, device=device)

    # runAll_division(datasets='DBP15K/fr_en(dbp15)/', model_type='rel_path', is_cuda=is_cuda, device=device)
    runAll_division(datasets='WN31/EN_DE_15K_V2/', model_type='rel_path', outname=outname, n_layers=n_layers, is_cuda=is_cuda, device=device)
    runAll_division(datasets='WN31/EN_DE_15K_V1/', model_type='rel_path', outname=outname, n_layers=n_layers, is_cuda=is_cuda, device=device)


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    is_cuda = False
    device = "2"
    print("GPU No: {}".format(device))

    # DBP15K/fr_en(dbp15) ja_en(dbp15), zh_en(dbp15)
    datasets = 'WN31/EN_DE_15K_V1/' # EN_DE_15K_V1、EN_FR_15K_V1
    #datasets = 'DWY100K/dbp_yg/' # DWY100K/dbp_wd, DWY100K/dbp_yg
    setRunmodel(datasets=datasets, division='1/', outname='tt', model_type='rel', is_cuda=is_cuda, device=device)

################

