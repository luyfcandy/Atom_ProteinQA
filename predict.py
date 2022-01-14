import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR))

import math
import glob
import torch
import argparse
import random
import importlib
import pickle
from tqdm import tqdm

import numpy as np
import time

from utils.logger import setup_logger
from checkpoint.config import cfg, cfg_from_yaml_file
from checkpoint.pc_graph_modelatomfuse import ProteinGCN, PCGCNFusion

aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
atom_list = ['N', 'C', 'O', 'S', 'H', 'X']

def load_checkpoint(args, model):
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    logger.info("=> loaded successfully '{}'".format(args.ckpt_path))
    del checkpoint
    torch.cuda.empty_cache()

def load_data(file):
    with open(file,'rb') as f:
        xyz = pickle.load(f)  # load xyz data
        aa_feature = pickle.load(f)  # load amino acid feature
        atom_feature = pickle.load(f)  # load atom feature
        residue_idx = pickle.load(f)  # load residue index
        nbr_info = pickle.load(f)  # edge features for each atom
        nbr_fea_idx = pickle.load(f)  # edge feature indexes

    return {'xyz': xyz,
            'aa_feature': aa_feature,
            'atom_feature': atom_feature,
            'residue_idx': residue_idx,
            'edge_feature': nbr_info,
            'edge_idx': nbr_fea_idx,
            'name': file}


def process_input_data(data_dict, config, vote_num=3):
    xyz = data_dict['xyz']
    aa_feature = data_dict['aa_feature']
    atom_feature = data_dict['atom_feature']
    aa_one_hot = np.zeros((len(aa_feature), 21))
    atom_one_hot = np.zeros((len(atom_feature), 6))
    for i, feat in enumerate(aa_feature):
        aa_one_hot[i, aa_feature[i]] = 1
        atom_one_hot[i, atom_feature[i]] = 1
    feature = np.concatenate([aa_one_hot, atom_one_hot], 1)

    coords = []
    features = []

    # divide by voxel size
    for v in range(vote_num):
        coord = np.ascontiguousarray(xyz - xyz.mean(0))
        m = np.eye(3) + np.random.randn(3, 3) * 0.1
        m[0][0] *= np.random.randint(0, 2) * 2 - 1
        m /= config.voxel_size

        # rotation (currently only on z-axix)
        theta = np.random.rand() * 2 * math.pi
        axis_id = random.choice([0, 1, 2])
        if axis_id == 0:  # rotate along x-axis
            m = np.matmul(m, [[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])
        if axis_id == 1:  # rotate along y-axis
            m = np.matmul(m, [[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
        if axis_id == 2:  # rotate along z-axis
            m = np.matmul(m, [[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        #m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        coord = np.matmul(coord, m)

        # place on (0,0) and crop out the voxel outside the full_scale
        m = coord.min(0)
        M = coord.max(0)
        offset = - m + np.clip(config.full_scale - M + m - 0.001, 0, None) * np.random.rand(3) + \
                 np.clip(config.full_scale - M + m + 0.001, None, 0) * np.random.rand(3)

        coord += offset
        coord = torch.Tensor(coord).long()
        coords.append(torch.cat([coord, torch.LongTensor(coord.shape[0], 1).fill_(v)], 1))
        features.append(torch.Tensor(feature))

    inputs = {
        'coords': torch.cat(coords, 0),
        'features': torch.cat(features, 0)
    }

    return inputs

def atom_to_residue(atom_lddt,res_index):
    s_len=int(res_index[-1])
    res_index=np.array(res_index)
    res_index=res_index-1
    new_pool=[[] for ii in range(s_len)]
    reduce_array=np.zeros((s_len))
    val_position = np.zeros(s_len)
    for rid in range(len(res_index)):
        res_idx=int(res_index[rid])
        new_pool[res_idx].append(atom_lddt[rid])
        val_position[res_idx] = 1
    for rid2 in range(s_len):
        tmp_arr=np.array(new_pool[rid2])
        new_arr=np.mean(tmp_arr)
        reduce_array[rid2]=new_arr
    valid_idx = np.where(val_position == 1)
    res_lddt_part = reduce_array[valid_idx]
    return res_lddt_part

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
            self.var = var
    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='6', help='Specify gpu devices')
    parser.add_argument('--data_path', type=str, default='./examples/pkls/', help='Data folder containing pkl files')
    parser.add_argument('--xyz_path', type=str, default='./examples/xyz/', help='Data folder containing xyz files')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/ckpt_8k2084_93.pth', help='Path of checkpoint')
    parser.add_argument('--ckpt_gcns1', type=str, default='./checkpoint/ckpt_decoy8k_gcns1.pth',
                        help='Path of gcns1 checkpoint')
    parser.add_argument('--ckpt_gcns2', type=str, default='./checkpoint/ckpt_decoy8k_gcns2.pth',
                        help='Path of gcns2 checkpoint')
    parser.add_argument('--ckpt_gcnall', type=str, default='./checkpoint/ckpt_8k2084_gcn5.pth',
                        help='Path of gcns2 checkpoint')
    parser.add_argument('--cfg_path', type=str, default='./checkpoint/SparseConv_AtomR.yaml', help='Path of config')
    parser.add_argument('--log_dir', type=str, default='./examples/prediction/', help='Prediction folder')
    parser.add_argument('--filename_suffix', type=str, default='atom_lddt', help='Filename suffix to save')
    parser.add_argument('--vote_num', type=int, default=3, help='Voting number')
    parser.add_argument('--save_atomlddt', action='store_true', default=True,
                        help='save atom-level in a txt file separately')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs(args.log_dir, exist_ok=True)
    atom_dir = os.path.join(args.log_dir.strip(), "atom_lddt")
    res_dir = os.path.join(args.log_dir.strip(), "all_lddt")
    os.makedirs(atom_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    logger = setup_logger(output=args.log_dir + '/log_test.txt', name='testing')

    # load configuration
    cfg_file = args.cfg_path
    cfg_from_yaml_file(cfg_file, cfg)
    logger.info(cfg)

    # set random seed
    random.seed(cfg.manualSeed)
    torch.manual_seed(cfg.manualSeed)
    np.random.seed(cfg.manualSeed)
    torch.cuda.manual_seed(cfg.manualSeed)

    # model
    logger.info('Load Model...')
    f_model = importlib.import_module('checkpoint.' + 'sparseconvunet')
    print('device: ', torch.cuda.current_device())
    model_pc = f_model.get_model(cfg).cuda()
    load_checkpoint(args, model_pc)

    # load all model args from pretrained model

    # build model
    kwargs = {
        'h_a': 64,  # Dim of the hidden atom embedding learnt
        'h_g': 32,  # Dim of the hidden graph embedding after pooling
        'n_conv': 4,  # Number of GCN layers
        'random_seed': 123,  # Seed to fix the simulation
        'lr': 0.001,  # Learning rate for optimizer
    }

    h_b = 43
    kwargs['h_b'] = h_b

    model_gcns1 = ProteinGCN(**kwargs)
    model_gcns1 = torch.nn.DataParallel(model_gcns1)
    model_gcns1.cuda()
    model_gcns2 = PCGCNFusion(**kwargs)
    model_gcns2 = torch.nn.DataParallel(model_gcns2)
    model_gcns2.cuda()

    if args.ckpt_gcnall is not None and os.path.isfile(args.ckpt_gcnall):
        checkpointall = torch.load(args.ckpt_gcnall, map_location='cpu')
        from_epoch = checkpointall['epoch'] + 1
        model_gcns1.module.load_state_dict(checkpointall['state_dict1'])
        model_gcns1.module.optimizer.load_state_dict(checkpointall['optimizer1'])
        model_gcns2.module.load_state_dict(checkpointall['state_dict2'])
        model_gcns2.module.optimizer.load_state_dict(checkpointall['optimizer2'])
        print("loaded model '{}' successfully".format(args.ckpt_gcnall))
    else:
        print("no model_gcn found at '{}'".format(args.ckpt_gcnall))


    # load data
    files=[]
    test_file_names=os.listdir(args.data_path)
    for item in test_file_names:
        if(item[-4:] != '.pkl'):
            continue
        files.append(args.data_path+item)
    num_files = len(files)
    print('decoy number: ', num_files)
    model_pc.eval()
    model_gcns1.eval()
    model_gcns2.eval()
    skip_list=[]
    import bad_ids_casp
    skip_list=bad_ids_casp.skip_casp13_8k
    tall_s=time.time()
    gdf = GaussianDistance(dmin=0, dmax=15, step=0.4)

    for f in tqdm(files):
        times=time.time()
        file_name = f[f.rfind('/') + 1:]
        print(file_name[:-4])
        protein_name = file_name[:file_name.find('_')]
        chain_name = file_name
        if os.path.exists(os.path.join(atom_dir, file_name[:-4]+'.atom_lddt')):
            continue
        if (file_name[:-4] in skip_list):
            continue

        data_dict = load_data(f)
        residue_idx = data_dict['residue_idx']

        vote_pool_regress = torch.zeros(data_dict['xyz'].shape[0], 1)
        point_idx = torch.arange(data_dict['xyz'].shape[0])
        point_idx = point_idx.repeat(args.vote_num)
        data = process_input_data(data_dict, cfg, vote_num=args.vote_num)
        torch.cuda.synchronize()

        for k in data.keys():
            if k in ['features', 'reg_labels']:
                data[k] = data[k].cuda()

        # get output
        end_points, shallow_feat, deep_feat = model_pc(data)
        if 'regression' in cfg:
            vote_pool_regress = vote_pool_regress.index_add_(0, point_idx, end_points['residual'].cpu())
            vote_pool_regress /= args.vote_num
            vote_pool_regress = vote_pool_regress.squeeze()
            prediction = vote_pool_regress
        else:
            raise TypeError('Please only set `regression` in yaml fime!')
        pcpred_list=prediction.detach().numpy()
        large_pos = np.where(pcpred_list > 1)
        small_pos = np.where(pcpred_list < 0)
        pcpred_list[large_pos] = 1
        pcpred_list[small_pos] = 0

        #shallow_feat = (shallow_feat+25)/25.0-1

        edge_npy = data_dict['edge_feature']
        edge_feat = torch.Tensor(np.concatenate([gdf.expand(edge_npy[:, :, 0]), edge_npy[:, :, 1:]], axis=2))
        edge_indexes = torch.LongTensor(data_dict['edge_idx'])
        pc_points = int(shallow_feat.shape[0]/args.vote_num)
        edge_num = edge_feat.shape[0]
        amino_idx_torch = torch.Tensor(residue_idx)-1
        shallow_feat_batch = torch.zeros(args.vote_num, edge_num, shallow_feat.shape[1])
        deep_feat_batch = torch.zeros(args.vote_num, edge_num, deep_feat.shape[1])
        atom_mask_batch = torch.ones((args.vote_num, edge_num), dtype = torch.long)
        amino_idx_batch = torch.zeros(args.vote_num, edge_num)
        atom_feat_batch = torch.zeros(args.vote_num, edge_num)
        edge_feat_batch = torch.zeros(args.vote_num, edge_num, edge_feat.shape[1], edge_feat.shape[2])
        edge_idx_batch = torch.zeros(args.vote_num, edge_num, edge_feat.shape[1])
        for vid in range(args.vote_num):
            shallow_feat_batch[vid] = shallow_feat[vid*pc_points:vid*pc_points+edge_num]
            deep_feat_batch[vid] = deep_feat[vid * pc_points:vid * pc_points + edge_num]
            amino_idx_batch[vid] = amino_idx_torch[:edge_num]
            edge_feat_batch[vid] = edge_feat
            edge_idx_batch[vid] = edge_indexes
        edge_idx_batch=edge_idx_batch.long()

        input_s1 = [edge_feat_batch.cuda(), edge_idx_batch.cuda(),
                    atom_mask_batch.cuda(), shallow_feat_batch.cuda()]
        input_s2 = [atom_mask_batch.cuda(), deep_feat_batch.cuda()]

        predicted = model_gcns1(input_s1)
        graph_fea = predicted[3].cpu().detach()
        gcnout1 = predicted[1]
        pred_list1 = torch.mean(gcnout1, dim=0)
        pred_list1 = pred_list1.squeeze().cpu().detach().numpy()
        graph_fea = graph_fea.cuda()
        predictedf = model_gcns2(input_s2, graph_fea)
        pred_atom = predictedf[0]
        prediction = torch.mean(pred_atom, dim=0)

        pred_list=prediction.squeeze().cpu().detach().numpy()
        #global_lddt=np.mean(pred_list)
        if (len(pred_list) < len(residue_idx)):
            for i in range(len(residue_idx) - len(pred_list)):
                pred_list = np.append(pred_list, pred_list[-1])
                pred_list1 = np.append(pred_list1, pred_list1[-1])
        avg_pred_list = (pred_list1+pred_list)/2.0
        #avg_pred_list = pred_list
        pred_list = avg_pred_list
        residue_lddt=atom_to_residue(pred_list, residue_idx)
        global_lddt = np.mean(residue_lddt)
        pred_dict={}
        pred_dict['atom_lddt']=pred_list.tolist()
        pred_dict['residue_lddt']=residue_lddt
        pred_dict['global_lddt']=global_lddt
        np.save(os.path.join(res_dir, file_name[:-4]+'.npy'), pred_dict)

        if args.save_atomlddt:
            xyz_file=args.xyz_path+file_name[:-4]+'.xyz'
            save_data = np.loadtxt(xyz_file, dtype=str)
            save_data[:, -1] = pred_list
            np.savetxt(os.path.join(atom_dir, file_name[:-4]+'.atom_lddt'),
                   save_data, fmt='%s')
        torch.cuda.synchronize()

        timee = time.time()
        logger.info('Decoy: %s, Global lddt: %.4f, Time: %.4f' % (
            file_name[:-4], global_lddt, timee-times))
    tall_e = time.time()
    all_cost=tall_e-tall_s
    logger.info('Total time: %.5f' % (all_cost))








