import os

from ddpg_per import DDPG

from envs import TradingEnv

if __name__ == "__main__":

    # disable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID'

    # specify what to test
    delta_action_test = True
    bartlett_action_test = False

    # set init_ttm, spread, and other parameters according to the env that the model is trained
    env_test = TradingEnv(continuous_action_flag=True, sabr_flag=False, dg_random_seed=5, spread=0.001, num_contract=1, init_ttm=20, trade_freq=1, num_sim=1001)
    ddpg_test = DDPG(env_test)

    ddpg_test.test(1001, delta_flag=delta_action_test, bartlett_flag=bartlett_action_test)