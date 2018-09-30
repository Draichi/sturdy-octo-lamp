import sys, gym
sys.path.append("trading_env")
from tensorflow.python.keras.layers import Input, Dense, Activation
from drlbox.trainer import make_trainer
from drlbox.evaluator import make_evaluator
# from baselines import deepq
from tesr import *

if __name__ == '__main__':
    evaluator = make_evaluator(
        env_maker=lambda: gym.make("EuroDolEval-v0"),
        render_timestep=1,
        render_end=False,
        num_episodes=1,
        algorithm='acer',
        load_model="dir/model.h5",
        verbose=True,
        )
    evaluator.run()