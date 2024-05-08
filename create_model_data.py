import torch
import numpy as np

import sys, os

from torch.utils.data import DataLoader
from types import SimpleNamespace
import torch.nn as nn
import argparse

sys.path.append(os.path.join('../',os.path.dirname(os.path.abspath(''))))
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
print(sys.path)

from train_GNODE_multiple import MDSimNet

num_particles = 258
BOX_SIZE = 27.27
box_size = np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE])

def network_trajectory(start_pos, start_vel, t_size, t_howmany, special, architecture, augmentation, pretrained, cp_name, epoch):
    PATH = f'./model_ckpt/{cp_name}/checkpoint_{epoch}.ckpt' ##replace with your model directory
    SCALER_CKPT = f'./model_ckpt/{cp_name}/scaler_{epoch}.npz'
    args = SimpleNamespace(use_layer_norm=False,
                        encoding_size=128,
                        hidden_dim=128,
                        edge_embedding_dim=128,
                        drop_edge=True,
                        conv_layer=3,
                        rotate_aug=False,
                        update_edge=False,
                        use_part=False,
                        data_dir='',
                        mode = 'test',
                        architecture = architecture,
                        augmentation = augmentation,
                        pretrained = pretrained,
                        loss='mae')
    model = MDSimNet.load_from_checkpoint(PATH, args=args)
    model.load_training_stats(SCALER_CKPT)
    model.cuda()
    model.eval()

    with torch.no_grad():

        trajectory, temp = model.run_the_network(start_pos, start_vel, t_size, t_howmany, special)


        trajectory = np.stack(trajectory, axis=0)

    return trajectory, temp

def save_trajectories_and_temp(args):
    t_size = 1
    t_howmany = 20
    cp_name = args.cp_name
    epoch = args.epoch
    architecture = args.architecture
    which = "first"
    #which = 'second'

    directory = os.path.join('Results_dir',f'results_{cp_name}_epoch={epoch}_{which}')
    os.makedirs(directory, exist_ok=True)

    trajectory_real = []
    trajectory_model = []
    trajectory_model_01 = []
    trajectory_model_02 = []
    trajectory_model_05 = []
    trajectory_model_1 = []
    trajectory_model_2 = []
    trajectory_model_3 = []
    trajectory_model_4 = []
    trajectory_model_5 = []
    trajectory_model_10 = []
    trajectory_model_15 = []
    trajectory_model_20 = []

    temp_model_01 = []
    temp_model_02 = []
    temp_model_05 = []
    temp_model_1 = []
    temp_model_2 = []
    temp_model_3 = []
    temp_model_4 = []
    temp_model_5 = []
    temp_model_10 = []
    temp_model_15 = []
    temp_model_20 = []

    for i in range (1000,1500):
        all = np.load(f'md_dataset/lj_data_20ts_to_test/data_0_{i}.npz')
        pos_one = all['pos']
        vel_one = all['vel']
        trajectory_real.append(pos_one)

        trajectory, temp = network_trajectory(pos_one, vel_one, t_size, t_howmany, True, architecture, cp_name, epoch)
        trajectory_model.append(trajectory)
        trajectory_model_01.append(trajectory[0])
        temp_model_01.append(temp[0])
        trajectory_model_02.append(trajectory[1])
        temp_model_02.append(temp[1])
        trajectory_model_05.append(trajectory[2])
        temp_model_05.append(temp[2])
        trajectory_model_1.append(trajectory[3])
        temp_model_1.append(temp[3])
        trajectory_model_2.append(trajectory[4])
        temp_model_2.append(temp[4])
        trajectory_model_3.append(trajectory[5])
        temp_model_3.append(temp[5])
        trajectory_model_4.append(trajectory[6])
        temp_model_4.append(temp[6])
        trajectory_model_5.append(trajectory[7])
        temp_model_5.append(temp[7])
        trajectory_model_10.append(trajectory[12])
        temp_model_10.append(temp[12])
        trajectory_model_15.append(trajectory[17])
        temp_model_15.append(temp[17])
        trajectory_model_20.append(trajectory[-1])
        temp_model_20.append(temp[-1])

        if ((i+1)%100 == 0): print(f"I'm done with {i-1000}")

    np.save(f'{directory}/trajectory_01.npy', np.stack(trajectory_model_01, axis=0))
    np.save(f'{directory}/temp_01.npy', np.stack(temp_model_01, axis=0))
    np.save(f'{directory}/trajectory_02.npy', np.stack(trajectory_model_02, axis=0))
    np.save(f'{directory}/temp_02.npy', np.stack(temp_model_02, axis=0))
    np.save(f'{directory}/trajectory_05.npy', np.stack(trajectory_model_05, axis=0))
    np.save(f'{directory}/temp_05.npy', np.stack(temp_model_05, axis=0))
    np.save(f'{directory}/trajectory_1.npy', np.stack(trajectory_model_1, axis=0))
    np.save(f'{directory}/temp_1.npy', np.stack(temp_model_1, axis=0))
    np.save(f'{directory}/trajectory_2.npy', np.stack(trajectory_model_2, axis=0))
    np.save(f'{directory}/temp_2.npy', np.stack(temp_model_2, axis=0))
    np.save(f'{directory}/trajectory_3.npy', np.stack(trajectory_model_3, axis=0))
    np.save(f'{directory}/temp_3.npy', np.stack(temp_model_3, axis=0))
    np.save(f'{directory}/trajectory_4.npy', np.stack(trajectory_model_4, axis=0))
    np.save(f'{directory}/temp_4.npy', np.stack(temp_model_4, axis=0))
    np.save(f'{directory}/trajectory_5.npy', np.stack(trajectory_model_5, axis=0))
    np.save(f'{directory}/temp_5.npy', np.stack(temp_model_5, axis=0))
    np.save(f'{directory}/trajectory_10.npy', np.stack(trajectory_model_10, axis=0))
    np.save(f'{directory}/temp_10.npy', np.stack(temp_model_10, axis=0))
    np.save(f'{directory}/trajectory_15.npy', np.stack(trajectory_model_15, axis=0))
    np.save(f'{directory}/temp_15.npy', np.stack(temp_model_15, axis=0))
    np.save(f'{directory}/trajectory_20.npy', np.stack(trajectory_model_20, axis=0))
    np.save(f'{directory}/temp_20.npy', np.stack(temp_model_20, axis=0))

    print("Finished saving yey")

def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--cp_name', default='model_ckpt/multiple')
    parser.add_argument('--epoch', default=4, type=int)
    parser.add_argument('--architecture', default='assisted', type=str)
    parser.add_argument('--augmentation', default=False)
    parser.add_argument('--pretrained', default=False)
    args = parser.parse_args()
    save_trajectories_and_temp(args)


if __name__ == '__main__':
    main()