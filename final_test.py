import numpy as np
import matplotlib.pyplot as plt

import sys, os

import numpy as np
from types import SimpleNamespace
import argparse

from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join('../',os.path.dirname(os.path.abspath(''))))
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
print(sys.path)

num_particles = 258
BOX_SIZE = 27.27
box_size = np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE])

def rdf_func(coords, box_size, dr=0.1, r_max=None):
    num_particles = len(coords)
    if r_max is None:
        r_max = np.min(box_size) / 2.0  # Use half the minimum box size as a default

    radii = np.arange(0, r_max + dr, dr)
    hist, _ = np.histogram(np.linalg.norm(coords - coords[:, np.newaxis], axis=-1), bins=radii)
    
    # Initialize g_r with zeros
    g_r = np.zeros_like(hist, dtype=float)

    # Correct for periodic boundary conditions
    tree = cKDTree(coords, boxsize=box_size)
    for i in range(num_particles):
        # Find neighbors considering PBC
        neighbors = tree.query_ball_point(coords[i], r_max, p=np.inf)
        
        for j in neighbors:
            if i != j:
                delta_r = coords[j] - coords[i]
                delta_r = np.where(np.abs(delta_r) > 0.5 * box_size, box_size - np.abs(delta_r), delta_r)
                distance = np.linalg.norm(delta_r)
                bin_index = np.digitize([distance], radii)[0] - 1  # Adjust for 0-based indexing
                if 0 <= bin_index < len(g_r):
                    g_r[bin_index] += 1

    # Normalize by the number of particles and the volume of each bin
    volume = 4/3 * np.pi * (radii[1:]**3 - radii[:-1]**3)
    g_r /= num_particles * volume

    return g_r, radii[:-1]  # Exclude the last bin edge for plotting

## for one time snapshot
def rdf_graph_one_snapshot(trajectory_real_np, trajectory_model, trajectory_GAMD, directory, epoch, femto):

    trajectory_model_np = np.concatenate(trajectory_model, axis = 0)
    trajectory_GAMD_np = np.stack(trajectory_GAMD, axis = 0)

    g_r_real_all = []
    g_r_model_all = []

    trajectory_real_np = np.mod(trajectory_real_np,BOX_SIZE)

    for i in range (trajectory_real_np.shape[0]):
        # Calculate RDF with periodic boundary conditions
        g_r_real, radii_real = rdf_func(trajectory_real_np[i], box_size, dr=0.1)
        g_r_real_all.append(g_r_real)

    g_r_real_average = np.mean(g_r_real_all, axis=0)

    trajectory_model_np = np.mod(trajectory_model_np,BOX_SIZE)

    for j in range(trajectory_model_np.shape[0]):
        # Calculate RDF with periodic boundary conditions
        g_r_model, radii_model = rdf_func(trajectory_model_np[j], box_size, dr=0.1)
        g_r_model_all.append(g_r_model)  

    g_r_model_average = np.mean(g_r_model_all, axis=0)

    plt.figure()
    plt.plot(radii_real, g_r_real_average, label='MD Simulation Data', color='cornflowerblue')
    plt.plot(radii_model, g_r_model_average, label='Model Data', color='darkorange')
    plt.xlabel('r')
    plt.ylabel('RDF(r)')
    plt.legend()
    plt.xlim(0, 14)
    plt.ylim(0, 0.05)
    plt.savefig(f"{directory}/rdfgraph_noGAMD_(femto={femto}, epoch={epoch}).png")
    plt.close()

    g_r_GAMD_all = []

    trajectory_GAMD_np = np.mod(trajectory_GAMD_np,BOX_SIZE)

    for i in range (trajectory_GAMD_np.shape[0]):
        # Calculate RDF with periodic boundary conditions
        g_r_GAMD, radii_GAMD = rdf_func(trajectory_GAMD_np[i], box_size, dr=0.1)
        g_r_GAMD_all.append(g_r_GAMD)

    g_r_GAMD_average = np.mean(g_r_GAMD_all, axis=0)

    plt.figure()
    plt.plot(radii_real, g_r_real_average, label='MD Simulation Data', color='cornflowerblue')
    plt.plot(radii_model, g_r_model_average, label='Model Data', color='darkorange')
    plt.plot(radii_GAMD, g_r_GAMD_average, label='GAMD Data', color='forestgreen')
    plt.xlabel('r')
    plt.ylabel('RDF(r)')
    plt.legend()
    plt.xlim(0, 14)
    plt.savefig(f"{directory}/rdfgraph_withGAMD_(femto={femto}, epoch={epoch}).png")
    plt.close()

    max_model, min_model = find_the_numbers(radii_model, g_r_model_average)
    max_real, min_real = find_the_numbers(radii_real, g_r_real_average)
    max_GAMD, min_GAMD = find_the_numbers(radii_GAMD, g_r_GAMD_average)

    relative_error_max_model_real = get_absolute_relative_error(max_model, max_real)
    relative_error_min_model_real = get_absolute_relative_error(min_model, min_real)
    relative_error_max_GAMD_real = get_absolute_relative_error(max_GAMD, max_real)
    relative_error_min_GAMD_real = get_absolute_relative_error(min_GAMD, min_real)
    relative_error_max_model_GAMD = get_absolute_relative_error(max_model, max_GAMD)
    relative_error_min_model_GAMD = get_absolute_relative_error(min_model, min_GAMD)

    dist_diff_model_real = calculate_distribution_difference(radii_real, g_r_model_average, g_r_real_average)
    dist_diff_GAMD_real = calculate_distribution_difference(radii_real, g_r_GAMD_average, g_r_real_average)
    dist_diff_model_GAMD = calculate_distribution_difference(radii_real, g_r_model_average, g_r_GAMD_average)

    file_path = os.path.join(directory, f"metrics_traj_femto={femto}" + '.txt')

    with open(file_path, 'w') as file:
        file.write(f"Max_model:{max_model}\n")
        file.write(f"Min_model:{min_model}\n")
        file.write(f"Max_real:{max_real}\n")
        file.write(f"Min_real:{min_real}\n")
        file.write(f"Max_GAMD:{max_GAMD}\n")
        file.write(f"Min_GAMD:{min_GAMD}\n")

        file.write(f"relative_error_max_model_real:{relative_error_max_model_real}\n")
        file.write(f"relative_error_min_model_real:{relative_error_min_model_real}\n")
        file.write(f"relative_error_max_GAMD_real:{relative_error_max_GAMD_real}\n")
        file.write(f"relative_error_min_GAMD_real:{relative_error_min_GAMD_real}\n")
        file.write(f"relative_error_max_model_GAMD:{relative_error_max_model_GAMD}\n")
        file.write(f"relative_error_min_model_GAMD:{relative_error_min_model_GAMD}\n")

        file.write(f"dist_diff_model_real:{dist_diff_model_real}\n")
        file.write(f"dist_diff_GAMD_real:{dist_diff_GAMD_real}\n")
        file.write(f"dist_diff_model_GAMD:{dist_diff_model_GAMD}\n")
        

def find_the_numbers(radii, g_r):
    index_of_max = np.argmax(g_r)
    max_radii = radii[index_of_max]
    index_of_min = index_of_max + np.argmin(g_r[index_of_max:])
    min_radii = radii[index_of_min]

    return max_radii, min_radii

def calculate_distribution_difference(radii, dist1, dist2):
    """
    Calculate the approximate integral of the absolute difference between two distributions
    sampled at the same radii points.

    Parameters:
    - radii: numpy array of radii at which distributions are sampled.
    - dist1: numpy array of the first distribution values at the sampled radii.
    - dist2: numpy array of the second distribution values at the sampled radii.

    Returns:
    - The approximate integral of the absolute difference between the two distributions.
    """

    # Calculate the absolute differences between the two distributions at each sampled point
    differences = np.abs(dist1 - dist2)

    # Assuming radii are equidistant, calculate the step size
    dr = np.mean(np.diff(radii))

    # Compute the approximate integral of the absolute differences
    integral = np.sum(differences) * dr

    return integral

def get_absolute_relative_error(num_model, num_real):
    return abs(num_model-num_real)/num_real


def save_xyz_from_numpy(coordinates, output_file):
    num_frames = coordinates.shape[0]
    num_atoms = coordinates.shape[1]

    with open(output_file, 'w') as f:
        for frame_idx in range(num_frames):
            f.write(str(num_atoms) + '\n')
            f.write("\n")
            for atom_idx in range(num_atoms):
                atom = coordinates[frame_idx, atom_idx]
                f.write(f"Ar {atom[0]} {atom[1]} {atom[2]}\n")

def get_temp_from_vel(vel):
    all_squared = np.sum(vel ** 2)
    coeff = 6.207563e-6 ##this is calculated
    return all_squared * coeff

def nice_temp_graphs(directory, femto, temp_real, temp_model_not_concat, temp_GAMD_not_concat):

    temp_model = np.concatenate(temp_model_not_concat)
    temp_GAMD = np.array(temp_GAMD_not_concat)

    indices = np.arange(len(temp_real))*20
    
    plt.figure()  # Start a new figure
    plt.plot(indices, temp_real, label='MD Simulation Data', color='cornflowerblue')  # MD Simulation Data in blue
    plt.plot(indices, temp_model, label='Model Data', color='darkorange')  # Model Data in red
    plt.xlabel('time')
    plt.ylabel('temperature')
    plt.ylim(0, 120)
    plt.legend()
    plt.savefig(f"{directory}/temp_graph_noGAMD_(femto={femto}).png")  # Save figure without GAMD data
    plt.close()  # Close the figure
    
    # Second figure: With GAMD data
    plt.figure()  # Start a new figure
    plt.plot(indices, temp_real, label='MD Simulation Data', color='cornflowerblue')  # MD Simulation Data in blue
    plt.plot(indices, temp_GAMD, label='GAMD Data', color='forestgreen')  # GAMD Data in orange
    plt.plot(indices, temp_model, label='Model Data', color='darkorange')  # Model Data in red
    plt.xlabel('time')
    plt.ylabel('temperature')
    plt.legend()
    plt.savefig(f"{directory}/temp_graph_withGAMD_(femto={femto}).png")  # Save figure with GAMD data
    plt.close()  # Close the figure
    

def do_it_all(directory, epoch, femto, pos_real, pos_GAMD, pos_model, temp_real, temp_GAMD, temp_model):
     rdf_graph_one_snapshot(pos_real, pos_model, pos_GAMD, directory, epoch, femto)
     nice_temp_graphs(directory, femto, temp_real, temp_model, temp_GAMD)
     output_file = os.path.join(directory, f'trajectory_GAMD_{femto}.xyz')
     pos_model = np.concatenate(pos_model, axis = 0)
     save_xyz_from_numpy(pos_model, output_file)
     pos_GAMD = np.stack(pos_GAMD, axis = 0)
     pos_GAMD = np.stack(pos_GAMD, axis = 0)
     save_xyz_from_numpy(pos_GAMD, output_file)


def test_main(args):
    t_size = 1
    t_howmany = 20
    cp_name = args.cp_name
    epoch = args.epoch

    directory = os.path.join('RESULTS_TEST_RESULTS_best_notmyGAMD_xyz',f'results_{cp_name}_epoch={epoch}')
    os.makedirs(directory, exist_ok=True)

    trajectory_real = []
    vel_real = []
    temp_real = []

    full_sequence = [0,0.1,0.2,0.5] + list(range(1,21))

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

    trajectory_GAMD_01 = []
    trajectory_GAMD_02 = []
    trajectory_GAMD_05 = []
    trajectory_GAMD_1 = []
    trajectory_GAMD_2 = []
    trajectory_GAMD_3 = []
    trajectory_GAMD_4 = []
    trajectory_GAMD_5 = []
    trajectory_GAMD_10 = []
    trajectory_GAMD_15 = []
    trajectory_GAMD_20 = []

    temp_GAMD_01 = []
    temp_GAMD_02 = []
    temp_GAMD_05 = []
    temp_GAMD_1 = []
    temp_GAMD_2 = []
    temp_GAMD_3 = []
    temp_GAMD_4 = []
    temp_GAMD_5 = []
    temp_GAMD_10 = []
    temp_GAMD_15 = []
    temp_GAMD_20 = []

    ## real data
    for i in range (1000,2000):
        all = np.load(f'md_dataset/lj_data_20ts_to_test/data_0_{i}.npz')
        pos_one = all['pos']
        vel_one = all['vel']
        trajectory_real.append(pos_one)
        vel_real.append(vel_one)
        temp_real.append(get_temp_from_vel(vel_one))

    trajectory_real = np.stack(trajectory_real, axis=0)[1:]
    temp_real_np = np.array(temp_real)

    print("MD Simulation data imported")

    ## GAMD data
    full_sequence_GAMD = [0.1,0.2,0.5, 1, 2, 3, 4, 5, 10, 15, 20]
    for i in range (1000, 2000):
        dir_now = f'md_dataset/lj_data_fromGAMD_notme_{i}'
        for j in full_sequence_GAMD:
            if (j>=1):
                all_file_name = f'data_{int(j*10)}.npz'
                where_exactly = os.path.join(dir_now,all_file_name)
                all = np.load(where_exactly)
                pos_one = all['pos']
                vel_one = all['vel']
                locals()[f'trajectory_GAMD_{j}'].append(pos_one)
                locals()[f'temp_GAMD_{j}'].append(get_temp_from_vel(vel_one))
            if (j<1):
                all_file_name = f'data_{int(j*10)}.npz'
                where_exactly = os.path.join(dir_now,all_file_name)
                all = np.load(where_exactly)
                pos_one = all['pos']
                vel_one = all['vel']
                locals()[f'trajectory_GAMD_{int(j*10):02}'].append(pos_one)
                locals()[f'temp_GAMD_{int(j*10):02}'].append(get_temp_from_vel(vel_one))

    print("GAMD data imported")
    
    ## model data
    full_sequence = [0.1,0.2,0.5, 1, 2, 3, 4, 5, 10, 15, 20]
    if (cp_name != 'GAMD_SONODE_20ts_for1_GAMDyes_freezeyes_PBCloss'):
        dir_to_get_model_from_first = os.path.join('Henna_best_trajectories_and_temps',f'results_{cp_name}_epoch={epoch}_first')
        dir_to_get_model_from_second = os.path.join('Henna_best_trajectories_and_temps',f'results_{cp_name}_epoch={epoch}_second')
        for j in full_sequence:
            if (j>=1):
                trajectory_file_name = f"trajectory_{j}.npy"
                file_path_first = os.path.join(dir_to_get_model_from_first, trajectory_file_name)
                data_first = np.load(file_path_first)
                locals()[f'trajectory_model_{j}'].append(data_first)
                file_path_second = os.path.join(dir_to_get_model_from_second, trajectory_file_name)
                data_second = np.load(file_path_second)
                locals()[f'trajectory_model_{j}'].append(data_second)
                locals()[f'trajectory_model_{j}'] = np.concatenate(locals()[f'trajectory_model_{j}'], axis=0)

                temp_file_name = f"temp_{j}.npy"
                file_path_first = os.path.join(dir_to_get_model_from_first, temp_file_name)
                data_first = np.load(file_path_first)
                locals()[f'temp_model_{j}'].append(data_first)
                file_path_second = os.path.join(dir_to_get_model_from_second, temp_file_name)
                data_second = np.load(file_path_second)
                locals()[f'temp_model_{j}'].append(data_second)
                locals()[f'temp_model_{j}'] = np.concatenate(locals()[f'temp_model_{j}'], axis=0)
            if (j<1):
                trajectory_file_name = f"trajectory_{int(j*10):02}.npy"
                file_path_first = os.path.join(dir_to_get_model_from_first, trajectory_file_name)
                data_first = np.load(file_path_first)
                locals()[f'trajectory_model_{int(j*10):02}'].append(data_first)
                file_path_second = os.path.join(dir_to_get_model_from_second, trajectory_file_name)
                data_second = np.load(file_path_second)
                locals()[f'trajectory_model_{int(j*10):02}'].append(data_second)

                temp_file_name = f"temp_{int(j*10):02}.npy"
                file_path_first = os.path.join(dir_to_get_model_from_first, temp_file_name)
                data_first = np.load(file_path_first)
                locals()[f'temp_model_{int(j*10):02}'].append(data_first)
                file_path_second = os.path.join(dir_to_get_model_from_second, temp_file_name)
                data_second = np.load(file_path_second)
                locals()[f'temp_model_{int(j*10):02}'].append(data_second)
                locals()[f'temp_model_{int(j*10):02}'] = np.concatenate(locals()[f'temp_model_{int(j*10):02}'], axis=0)

    else:
        dir_to_get_model_from = os.path.join('Henna_best_trajectories_and_temps',f'results_{cp_name}_epoch={epoch}')
        for j in full_sequence:
            if (j>=1):
                trajectory_file_name = f"trajectory_{j}.npy"
                file_path = os.path.join(dir_to_get_model_from, trajectory_file_name)
                data = np.load(file_path)
                locals()[f'trajectory_model_{j}'].append(data)
                locals()[f'trajectory_model_{j}'] = np.concatenate(locals()[f'trajectory_model_{j}'], axis=0)

                temp_file_name = f"temp_{j}.npy"
                file_path = os.path.join(dir_to_get_model_from, temp_file_name)
                data = np.load(file_path)
                locals()[f'temp_model_{j}'].append(data)
                locals()[f'temp_model_{j}'] = np.concatenate(locals()[f'temp_model_{j}'], axis=0)
            if (j<1):
                trajectory_file_name = f"trajectory_{int(j*10):02}.npy"
                file_path = os.path.join(dir_to_get_model_from, trajectory_file_name)
                data = np.load(file_path)
                locals()[f'trajectory_model_{int(j*10):02}'].append(data)
                locals()[f'trajectory_model_{int(j*10):02}'] = np.concatenate(locals()[f'trajectory_model_{int(j*10):02}'], axis=0)

                temp_file_name = f"temp_{int(j*10):02}.npy"
                file_path = os.path.join(dir_to_get_model_from, temp_file_name)
                data = np.load(file_path)
                locals()[f'temp_model_{int(j*10):02}'].append(data)
                locals()[f'temp_model_{int(j*10):02}'] = np.concatenate(locals()[f'temp_model_{int(j*10):02}'], axis=0)

    print("Model data imported")
    

    for j in full_sequence:
        femto = j*40
        if (j>=1): do_it_all(directory, epoch, femto, trajectory_real, locals()[f'trajectory_GAMD_{j}'], locals()[f'trajectory_model_{j}'], temp_real, locals()[f'temp_GAMD_{j}'], locals()[f'temp_model_{j}'])
        if (j<1): do_it_all(directory, epoch, femto, trajectory_real, locals()[f'trajectory_GAMD_{int(j*10):02}'], locals()[f'trajectory_model_{int(j*10):02}'], temp_real, locals()[f'temp_GAMD_{int(j*10):02}'], locals()[f'temp_model_{int(j*10):02}'])
        print(f"{femto}fs done :)")
        

def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--cp_name', default='GAMD_SONODE_20ts_for20_GAMDyes_freezeyes_PBCloss_AUG')
    parser.add_argument('--epoch', default=7, type=int)
    args = parser.parse_args()
    test_main(args)


if __name__ == '__main__':
    main()
