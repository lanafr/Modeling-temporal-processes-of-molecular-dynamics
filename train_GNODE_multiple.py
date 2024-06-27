###########################
# The foundation of this code is built upon GAMD (https://arxiv.org/abs/2112.03383, Li, Zijie and Meidani, Kazem and Yadav, Prakarsh and Barati Farimani, Amir, 2022.)
###########################

import argparse
import os, sys
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import jax
from pytorch_lightning.loggers import WandbLogger

import dgl.nn
import dgl.function as fn

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
#from nn_module import SimpleMDNetNew
from GAMD.graph_utils import NeighborSearcher, graph_network_nbr_fn
import time
from SIMPLER_GNN import * ##change thiss
from train_utils_seq import *

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# for water box
CUTOFF_RADIUS = 7.5
BOX_SIZE = 27.27

NUM_OF_ATOMS = 258

LAMBDA1 = 100.
LAMBDA2 = 1e-4
WANDB_DEBUG = True

os.environ["WANDB_MODE"] = "offline"

pos_var = 61.97
pos_mean = 13.635


def build_model(args, ckpt=None):

    param_dict = {
                  'encoding_size': args.encoding_size,
                  'out_feats': 3,
                  'hidden_dim': args.hidden_dim,
                  'edge_embedding_dim': args.edge_embedding_dim,
                  'conv_layer': 3,
                  'drop_edge': args.drop_edge,
                  'use_layer_norm': args.use_layer_norm,
                  'box_size': BOX_SIZE,
                  'architecture': args.architecture, ## simple or assisted
                  'augmentation': args.augmentation, ## true or false
                  'pretrained': args.pretrained, ## true or false
                  }

    print("Using following set of hyper-parameters")
    print(param_dict)
    model = EntireModel(**param_dict)

    if ckpt is not None:
        print('Loading model weights from: ', ckpt)
        model.load_state_dict((torch.load(ckpt)))
    return model


class MDSimNet(pl.LightningModule):
    def __init__(self, args, num_device=1, epoch_num=100, batch_size=1, learning_rate=3e-4, log_freq=1000,
                 model_weights_ckpt=None, scaler_ckpt=None):
        super(MDSimNet, self).__init__()
        self.pnet_model = build_model(args, model_weights_ckpt)
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_device = num_device
        self.log_freq = log_freq
        self.train_data_scaler_pos = StandardScaler()
        self.train_data_scaler_vel = StandardScaler()
        self.train_data_scaler_for = StandardScaler()
        self.training_mean_vel = np.array([0.])
        self.training_var_vel = pos_var# np.array([1.])
        self.training_mean_pos = pos_mean# np.array([0.])
        self.training_var_pos = np.array([1.])
        self.architecture = args.architecture

        if scaler_ckpt is not None:
            self.load_training_stats(scaler_ckpt)

        self.cutoff = CUTOFF_RADIUS
        self.nbr_searcher = NeighborSearcher(BOX_SIZE, self.cutoff)
        self.nbrlst_to_edge_mask = jax.jit(graph_network_nbr_fn(self.nbr_searcher.displacement_fn,
                                                                    self.cutoff,
                                                                    NUM_OF_ATOMS))
        self.nbr_cache = {}
        self.rotate_aug = args.rotate_aug
        self.data_dir = args.data_dir
        self.loss_fn = args.loss

        assert self.loss_fn in ['mae', 'mse']

    def load_training_stats(self, scaler_ckpt):
        if scaler_ckpt is not None:
            scaler_info = np.load(scaler_ckpt)
            self.training_mean_vel = scaler_info['mean_vel']
            self.training_var_vel = scaler_info['var_vel']
            self.training_mean_pos = scaler_info['mean_pos']
            self.training_var_pos = scaler_info['var_pos']
            #self.training_mean_for = scaler_info['mean_for']
            #self.training_var_for = scaler_info['var_for']

    
    def run_the_network(self, start_pos: np.ndarray, start_vel: np.ndarray, t_size, t_howmany, exact_test_times = True, verbose=False):  

        with torch.no_grad():

            temp_res = []
            start_vel = torch.from_numpy(start_vel).float().cuda()
            start_vel = self.scale([start_vel], self.train_data_scaler_vel).to('cuda').squeeze()      
            start_pos = torch.from_numpy(start_pos).float().cuda()
            #start_pos = self.scale([start_pos], self.train_data_scaler_pos).to('cuda').squeeze()
            start_pos = torch.squeeze(self.normalize_pos(start_pos)).to('cuda')

            if (exact_test_times == False):
                integration_time = (torch.arange(0, t_howmany+1) * t_size).to('cuda')
                integration_time = integration_time.float()
            if (exact_test_times == True):
                full_sequence = [0,0.1,0.2,0.5] + list(range(1,21))
                integration_time = torch.tensor(full_sequence, dtype=torch.float).to('cuda')

            pos_res, vel_res = self.pnet_model([start_pos],
                                [start_vel],
                                integration_time,
                                )

            pos_res = np.squeeze(self.denormalize_pos(pos_res.detach().cpu().numpy()))
            if (pos_res.shape[0]==258):
                pos_res = pos_res[np.newaxis, :, :]
            #vel_res = np.squeeze(self.denormalize_vel(vel_res.detach().cpu().numpy()))
            #pos_res = np.squeeze(pos_res.detach().cpu().numpy())
            notnor_vel = torch.tensor(np.squeeze(self.denormalize_vel(vel_res.detach().cpu().numpy()))).cuda()
            for i in range (notnor_vel.size()[0]):
                temp_res.append(self.get_temp_from_vel(notnor_vel[i]).detach().cpu().numpy())

        return pos_res, temp_res

    def make_a_graph(self, pos):
        edge_idx_tsr = self.search_for_neighbor(pos,
                                                    self.nbr_searcher,
                                                    self.nbrlst_to_edge_mask,
                                                    'all')

        center_idx = edge_idx_tsr[0, :]  # [edge_num, 1]
        neigh_idx = edge_idx_tsr[1, :]
        graph_now = dgl.graph((neigh_idx, center_idx))

        return graph_now

    def get_temp_from_vel(self, vel):
        all_squared = vel.pow(2).sum()
        coeff = 6.207563e-6 ## this is calculated
        return all_squared * coeff

    def scale(self, inp, scaler):
        scaled_sequence = []
        for seq in inp:
            seq = seq.squeeze()
            if seq.ndim == 1:
                seq = seq.reshape((-1, 1))  # Reshape 1D array to 2D
            b_pnum, dims = seq.shape
            seq = seq.detach().cpu()  # Move tensor to CPU
            scaler.partial_fit(seq.numpy())  # Convert to NumPy array
            scaled_seq = scaler.transform(seq.numpy())
            scaled_seq_tensor = torch.from_numpy(scaled_seq).float().view(b_pnum, dims)
            scaled_sequence.append(scaled_seq_tensor)
        return torch.stack(scaled_sequence)

    def normalize_pos(self, inp):
        normalized_sequence = []
        mean = torch.tensor(self.training_mean_pos, dtype=torch.float32).cuda()
        variance = torch.tensor(self.training_var_pos, dtype=torch.float32).cuda()
        std_dev = torch.sqrt(variance)

        for seq in inp:
            seq = seq.squeeze()
            if seq.ndim == 1:
                seq = seq.reshape((-1, 1))  # Reshape 1D array to 2D if necessary

            # Normalization
            normalized_seq = (seq - mean) / std_dev

            # Ensure the result is in the same shape and type as input
            normalized_seq = normalized_seq.view_as(seq)
            normalized_sequence.append(normalized_seq)

        return torch.stack(normalized_sequence)

    def denormalize_vel(self, vel):
        return self.denormalize(vel, self.training_var_vel, self.training_mean_vel)
    
    def denormalize_pos(self, pos):
        return self.denormalize(pos, self.training_var_pos, self.training_mean_pos)
    
    def denormalize_for(self, force):
        return self.denormalize(force, self.training_var_for, self.training_mean_for)

    def denormalize(self, normalized, var, mean):
        return normalized * \
                np.sqrt(var) +\
                    mean
        
    def periodic_difference(self, a, b, box_length):
        # Calculate periodic difference
        diff = a - b
        # Apply periodic boundary conditions
        diff -= box_length * torch.round(diff / box_length)
        return diff

    def calculate_periodic_mse_loss(self, pos_lst, pos_res, box_length):
        # Instantiate the MSE loss function
        mse_loss_fn = nn.MSELoss()
        
        # Calculate the periodic difference
        diff = self.periodic_difference(pos_res, pos_lst, box_length)
        
        # Adjust pos_res based on periodic difference for loss calculation
        adjusted_pos_res = pos_lst + diff
        
        # Calculate MSE loss
        mse_loss = mse_loss_fn(adjusted_pos_res, pos_lst)
        return mse_loss

    def training_step(self, batch, batch_nb):

        torch.cuda.empty_cache()

        #pos_lst = self.scale(batch[0]['pos'], self.train_data_scaler_pos).to('cuda')
        vel_lst = self.scale(batch[0]['vel'], self.train_data_scaler_vel).to('cuda')
        force_lst = self.scale(batch[0]['force'], self.train_data_scaler_for).to('cuda')
        pos_lst = self.normalize_pos(batch[0]['pos']).to('cuda')
        
        #pos_lst = torch.stack(batch[0]['pos'], dim=0).squeeze().to('cuda')
        #vel_lst = torch.stack(batch[0]['vel'], dim=0).squeeze().to('cuda')
        

        integration_time = torch.arange(pos_lst.size()[0]).to('cuda')
        integration_time = integration_time.float()


        start_pos = pos_lst[0]
        start_vel = vel_lst[0]
        
        BOX_SIZE = torch.tensor([27.27, 27.27, 27.27]).to('cuda')
        start_pos = torch.fmod(start_pos, BOX_SIZE) #this I added

        pos_res, vel_res = self.pnet_model([start_pos],
                                [start_vel],
                                integration_time,
                                )

        vel_lst = vel_lst[1:]
        pos_lst = pos_lst[1:]

        if self.loss_fn == 'mae':
            vel_loss = nn.L1Loss()(vel_lst, vel_res)
        else:
            vel_loss = nn.L1Loss(reduction='mean')(vel_lst, vel_res)

        box_length = 27.27
        cord_loss = self.calculate_periodic_mse_loss(pos_lst, pos_res, box_length)

        self.training_mean_vel = self.train_data_scaler_vel.mean_
        self.training_var_vel = self.train_data_scaler_vel.var_


        notnor_vel = torch.tensor(np.squeeze(self.denormalize_vel(vel_res.detach().cpu().numpy()))).cuda()
        notnor_orig_vel = torch.tensor(np.squeeze(self.denormalize_vel(vel_lst.detach().cpu().numpy()))).cuda()

        temp_predicted = self.get_temp_from_vel(notnor_vel[18])
        temp_actual = self.get_temp_from_vel(notnor_orig_vel[18])

        loss = vel_loss + cord_loss 

        self.log('cord_loss', cord_loss, on_step=True, prog_bar=True, logger=True)
        self.log('vel_loss', vel_loss, on_step=True, prog_bar=True, logger=True)
        self.log('temp_pred', temp_predicted, on_step=True, prog_bar=True, logger=True)
        self.log('temp actual', temp_actual, on_step=True, prog_bar=True, logger=True)
        self.log(f'actual loss:{self.loss_fn}', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
            optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            sched = StepLR(optim, step_size=5, gamma=0.001**(5/self.epoch_num))
            return [optim], [sched]

    def train_dataloader(self):

        dataset = sequence_of_pos(dataset_path=os.path.join(self.data_dir, 'lj_data_20ts'),
                               seed_num=10,
                               mode='train',
                               length=20)

        return DataLoader(dataset, batch_size=1, shuffle = True, pin_memory=False)
        

    def val_dataloader(self):

        dataset = sequence_of_pos(dataset_path=os.path.join(self.data_dir, 'lj_data_20ts'),
                               seed_num=10,
                               mode='test',
                               length=20)

        return DataLoader(dataset, batch_size=1, shuffle = True, pin_memory=False)


    def validation_step(self, batch, batch_nb):
        torch.cuda.empty_cache()

        with torch.no_grad():

            vel_lst = self.scale(batch[0]['vel'], self.train_data_scaler_vel).to('cuda')
            force_lst = self.scale(batch[0]['force'], self.train_data_scaler_for).to('cuda')
            pos_lst = self.normalize_pos(batch[0]['pos']).to('cuda')

            integration_time = torch.arange(pos_lst.size()[0]).to('cuda')
            integration_time = integration_time.float()


            start_pos = pos_lst[0]
            start_vel = vel_lst[0]
            
            BOX_SIZE = torch.tensor([27.27, 27.27, 27.27]).to('cuda')
            start_pos = torch.fmod(start_pos, BOX_SIZE) #this I added

            pos_res, vel_res = self.pnet_model([start_pos],
                                    [start_vel],
                                    integration_time,
                                    )
            vel_lst = vel_lst[1:]
            pos_lst = pos_lst[1:]

            if self.loss_fn == 'mae':
                vel_loss = nn.L1Loss()(vel_lst, vel_res)
            else:
                vel_loss = nn.MSELoss(reduction='mean')(vel_lst, vel_res)

            box_length = 27.27
            cord_loss = self.calculate_periodic_mse_loss(pos_lst, pos_res, box_length)

            self.log('val_cord_loss', cord_loss, on_step=True, prog_bar=True, logger=True, batch_size=1)
            self.log('val_vel_loss', vel_loss, on_step=True, prog_bar=True, logger=True, batch_size=1)

class ModelCheckpointAtEpochEnd(pl.Callback):
    """
       Save a checkpoint at epoch end
    """
    def __init__(
            self,
            filepath,
            save_step_frequency,
            architecture,
            prefix="checkpoint",
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
        """
        self.filepath = filepath
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.architecture = architecture
        self.just_name = os.path.basename(filepath)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: MDSimNet):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        if epoch % self.save_step_frequency == 0 or epoch == pl_module.epoch_num -1:
            filename = os.path.join(self.filepath, f"{self.prefix}_{epoch}.ckpt")
            scaler_filename = os.path.join(self.filepath, f"scaler_{epoch}.npz")

            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            np.savez(scaler_filename,
                        mean_vel=pl_module.training_mean_vel,
                        var_vel=pl_module.training_var_vel,
                        mean_pos=pl_module.training_mean_pos,
                        var_pos=pl_module.training_var_pos,
                        )
def train_model(args):
    lr = args.lr
    num_gpu = args.num_gpu
    check_point_dir = args.cp_dir
    min_epoch = args.min_epoch
    max_epoch = args.max_epoch
    weight_ckpt = args.state_ckpt_dir
    batch_size = args.batch_size
    architecture = args.architecture

    wandb_logger = WandbLogger(name="literallythehardestprojectever", project="MDNet", reinit=True)

    model = MDSimNet(epoch_num=max_epoch,
                                 num_device=num_gpu if num_gpu != -1 else 1,
                                 learning_rate=lr,
                                 model_weights_ckpt=weight_ckpt,
                                 batch_size=batch_size,
                                 args=args)

    cwd = os.getcwd()
    model_check_point_dir = os.path.join(cwd, check_point_dir)
    os.makedirs(model_check_point_dir, exist_ok=True)
    epoch_end_callback = ModelCheckpointAtEpochEnd(filepath=model_check_point_dir, save_step_frequency=5, architecture = architecture)
    checkpoint_callback = pl.callbacks.ModelCheckpoint()

    trainer = Trainer(devices=num_gpu, 
                      accelerator="gpu",
                      callbacks=[epoch_end_callback, checkpoint_callback],
                      min_epochs=min_epoch,
                      max_epochs=max_epoch,
                      #amp_backend='apex',
                      #amp_level='O1',
                      benchmark=True,
                      strategy='ddp_find_unused_parameters_true',
                      #strategy='ddp', #changed from distributed_backend
                      logger=wandb_logger,
                      )
    trainer.fit(model) #, ckpt_path = ...)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_epoch', default = 30, type=int)
    parser.add_argument('--max_epoch', default = 30, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--cp_dir', default='./model_ckpt/multiple')
    parser.add_argument('--state_ckpt_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--encoding_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--edge_embedding_dim', default=128, type=int)
    parser.add_argument('--drop_edge', action='store_true', default = True)
    parser.add_argument('--use_layer_norm', action='store_true', default = True)
    parser.add_argument('--disable_rotate_aug', dest='rotate_aug', default=True, action='store_false')
    parser.add_argument('--data_dir', default='./md_dataset')
    parser.add_argument('--loss', default='mae')
    parser.add_argument('--num_gpu', default=-1, type=int)
    parser.add_argument('--architecture', default='assisted', type=str) ## 'assisted' or 'simple'
    parser.add_argument('--augmentation', default = False, type=bool) ## 'true' or 'false'
    parser.add_argument('--pretrained', default= True, type=bool) ## 'true' or 'false'

    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()
