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
    print(observation_space)
    print("done printing observation_space")
    feature = inp_state
    for num_hid in num_hid_list:
        feature = Dense(num_hid)(feature)
        feature = Activation('relu')(feature)
    return inp_state, feature
# from keras.models import Sequential
# from keras.layers import *
# from keras.optimizers import *
# import keras
# from keras.layers import PReLU
# import ipdb
# from tensorflow.python.keras import layers, initializers, models
# def make_model(env):
#     input_shape = observation_space
#     #print(observation_space)
#     #input_shape = np.reshape(-1,1,67)
#     # input state
#     #pht_state = layers.Input(shape=input_shape)
#     original_matrix = np.genfromtxt("C:\\Users\\Teert\\AppData\\Roaming\\MetaQuotes\\Terminal\\6BCF14A3917E5BC71FD48812B8ED0586\\MQL4\\Files\\EURUSD,5.csv", delimiter=",", skip_header=39)
#     stateCnt = original_matrix.shape[1]
    # model = Sequential()
    # #print(self.actionCnt)
    # model.add(Conv1D(filters=32,kernel_size=1, dilation_rate=2, activation='relu',padding='valid', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.005), input_shape=(10, stateCnt)))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(kernel_size = (1), filters = 32, dilation_rate=4, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(kernel_size = (1), filters = 64, dilation_rate=8, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(kernel_size = (1), filters = 64, dilation_rate=16, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # model.add(MaxPooling1D())
    # model.add(Dropout(0.2))
    # model.add(Conv1D(kernel_size = (1), filters = 64, dilation_rate=32, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(kernel_size = (1), filters = 128, strides=2, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(kernel_size = (1), filters = 128, strides=2, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(kernel_size = (1), filters = 128, strides=2, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # model.add(Dropout(0.2))
    # model.add(Bidirectional(GRU(48,return_sequences=True),merge_mode='concat'))
    # #model.add(Conv1D(filters=100,kernel_size=3,activation='relu',padding='same',input_shape=(10, self.stateCnt)))
    # #model.add(Bidirectional(CuDNNGRU(48,return_sequences=True),merge_mode='concat'))
    # model.add(PReLU())
    # model.add(Flatten())
    # model.add(Dense(24, use_bias=False))
    # #model.add(LeakyReLU(alpha=0.1))
    # model.add(PReLU())
    # model.add(Dense(12, use_bias=False))
    # #model.add(LeakyReLU(alpha=0.1))
    # model.add(PReLU())
    # model.add(Dense(4, activation="linear"))
    # #adamw =  AdamW(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, weight_decay=0.025, batch_size=1, samples_per_epoch=1, epochs=1)
    # adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # model.compile(loss="mse" , optimizer="adam" , metrics=["accuracy"])

    # return model
#
# if __name__ == '__main__':
#     trainer = make_trainer(
#         algorithm='dqn',
#         env_maker=lambda: gym.make("EuroDolTrain-v0"),
#         model_maker=make_model,
#         num_parallel=1,
#         train_steps=1000,
#         online_learning=False,
#         verbose=True,
#         batch_size=1,
#         save_dir="F:\\model.h5"
#         )
#     trainer.run()

if __name__ == '__main__':
    trainer = make_trainer(
       algorithm='acer',
       env_maker=lambda: gym.make("EuroDolTrain-v0"),
       feature_maker=lambda obs_space: make_feature(obs_space, [200, 100]),
       num_parallel=1,
       train_steps=1000,
       online_learning=False,
       verbose=True,
       batch_size=1,
       save_dir="F:\\model.h5"
       )
    trainer.run()

    evaluator = make_evaluator(
        env_maker=lambda: gym.make("EuroDolEval-v0"),
        render_timestep=1,
        load_model="F:\\dir\\model.h5",
        render_end=False,
        num_episode=1,
        algorithm='acer',
        verbose=True,
        )
    evaluator.run()
