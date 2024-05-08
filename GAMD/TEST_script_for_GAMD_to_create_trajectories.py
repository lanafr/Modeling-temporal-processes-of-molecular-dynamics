import torch
from openmmtools import testsystems
from simtk.openmm import *
from simtk.openmm.app import *
import simtk.unit as unit
import logging
import numpy as np

from openmmtools.constants import kB
from openmmtools import respa, utils

from types import SimpleNamespace
import argparse

# Energy unit used by OpenMM unit system
_OPENMM_ENERGY_UNIT = unit.kilojoules_per_mole
from openmmtools import states, integrators
import time
import numpy as np
import sys
import os
from functools import partial

from hack_integrator import HackNoseHooverIntegrator, HackHalfNoseHooverIntegrator

from types import SimpleNamespace
from LJ.just_nice_clean_code.GAMD.FORCE_PREDICTOR_GAMD import ParticleNetLightning

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

logger = logging.getLogger(__name__)

def simulate_with_GAMD(pos_one, vel_one, i):
    BOX_SCALE = 2.0   # to match the scale in experimental data
    DT = 2.0

    platform = Platform.getPlatformByName('CPU')
    P_NUM = 258

    fluid = testsystems.LennardJonesFluid(nparticles=P_NUM, reduced_density=0.50, shift=True)
    [topology, system, positions] = fluid.topology, fluid.system, fluid.positions
    pos = pos_one
    vel = vel_one
    p_num = positions.shape[0]
    timestep = DT * unit.femtoseconds
    temperature = 100 * unit.kelvin
    chain_length = 10
    GAMMA = 25. / unit.picosecond
    dummy_integrator = CompoundIntegrator()
    integrator1 = HackNoseHooverIntegrator(system, temperature,
                                        collision_frequency=GAMMA,
                                        chain_length=chain_length,
                                        timestep=timestep)
    integrator2 = HackHalfNoseHooverIntegrator(system, temperature,
                                            collision_frequency=GAMMA,
                                            chain_length=chain_length,
                                            timestep=timestep)
    dummy_integrator.addIntegrator(integrator1)
    dummy_integrator.addIntegrator(integrator2)

    dummy_simulator = Simulation(topology, system, dummy_integrator, platform=platform)

    dummy_simulator.context.setPositions(pos*unit.angstrom)
    dummy_simulator.context.setVelocities(vel*(unit.meter / unit.second))

    # ===========================================================================
    NUM_OF_ATOMS = positions.shape[0]                  # (258*3)
    print(f'Simulating {NUM_OF_ATOMS} number of atoms')
    PATH = '/home/hpc/b118bb/b118bb12/000/0901/code_GAMD1/LJ/model_ckpt/GAMD_not_from_me/checkpoint.ckpt'
    SCALER_CKPT = '/home/hpc/b118bb/b118bb12/000/0901/code_GAMD1/LJ/model_ckpt/GAMD_not_from_me/scaler.npz'
    args = SimpleNamespace(use_layer_norm=False,
                        encoding_size=128,
                        hidden_dim=128,
                        edge_embedding_dim=128,
                        drop_edge=False,
                        conv_layer=4,
                        rotate_aug=False,
                        update_edge=False,
                        use_part=False,
                        data_dir='',
                        loss='mae')
    model = ParticleNetLightning.load_from_checkpoint(PATH, args=args)
    model.load_training_stats(SCALER_CKPT)
    model.cuda()
    model.eval()

    dataReporter = StateDataReporter(f'./log_nvt_gnn_nosehoover.txt', 100,
                                    totalSteps=int(100000//DT),
                                    step=True, time=True,
                                    kineticEnergy=True,
                                    temperature=True, separator='\t')
    dummy_simulator.reporters.append(dataReporter)
    dummy_simulator.minimizeEnergy(1e-6)

    dummy_state = dummy_simulator.context.getState(getPositions=True,
                                                getVelocities=True,
                                                getForces=True)
    pos = dummy_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    vel = dummy_state.getVelocities(asNumpy=True).value_in_unit((unit.meter / unit.second))

    force = model.predict_forces(pos)
    force = force*(unit.kilojoules_per_mole/unit.nanometers)

    #print(f'Using collision frequency: {GAMMA}')
    for t in range(int(400//DT)):
        if (t+1)%100 == 0:
            print(f'Finished {(t+1)} steps')

        dummy_integrator.setCurrentIntegrator(0)
        if t != 0:
            integrator1.copy_state_from_integrator(integrator2)
        integrator1.setPerDofVariableByName('force_last', force)
        dummy_simulator.step(1)

        dummy_state = dummy_simulator.context.getState(getPositions=True,
                                                       getVelocities=True,
                                                    enforcePeriodicBox=True)
        pos = dummy_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        vel = dummy_state.getVelocities(asNumpy=True).value_in_unit(unit.meter / unit.second)
        force = model.predict_forces(pos)
        force = force*(unit.kilojoules_per_mole/unit.nanometers)
        dummy_integrator.setCurrentIntegrator(1)
        integrator2.copy_state_from_integrator(integrator1)
        integrator2.setPerDofVariableByName('gnn_force', force)
        os.makedirs(f'./md_dataset/lj_data_notme_{i}/', exist_ok=True)
        dummy_simulator.step(1)

        np.savez(f'./md_dataset/lj_data_fromGAMD_{i}/data_{t}.npz',
                 pos=pos,
                 vel=vel,
                 forces=force)

def main():
    print("I'm here")
    for i in range(1800, 2000):
        all = np.load(f'md_dataset/lj_data_20ts_to_test/data_0_{i}.npz')
        pos_one = all['pos']
        vel_one = all['vel']
        simulate_with_GAMD(pos_one, vel_one, i)
        if ((i+1)%100==0): print (f"Done with {i}")

if __name__ == '__main__':
    main()