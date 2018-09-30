import sys, gym
sys.path.append("trading_env")
from tensorflow.python.keras.layers import Input, Dense, Activation
from drlbox.trainer import make_trainer
from drlbox.evaluator import make_evaluator
# from baselines import deepq
from tesr import *

def callback(lcl, _glb):
    return False
def make_feature(observation_space, num_hid_list):
    inp_state = Input(shape=observation_space)
    print('\n observation space:',observation_space)
    feature = inp_state
    for num_hid in num_hid_list:
        feature = Dense(num_hid)(feature)
        feature = Activation('relu')(feature)
    return inp_state, feature

if __name__ == '__main__':
    trainer = make_trainer(
        algorithm='impala',
        env_maker=lambda: gym.make("EuroDolTrain-v0"),
        feature_maker=lambda obs_space: make_feature(obs_space, [200, 100]),
        num_parallel=1,
        train_steps=2000,
        verbose=True,
        batch_size=2,
        save_dir="dir",
        )
    trainer.run()