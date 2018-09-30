PATH_TO_CSV = "EURUSD_5.csv"
PATH_TO_MODEL = "dir"
PATH_TO_LOAD_MODEL = "/home/lucas/Documents/algotrading/dir/model.h5"

# PATH_TO_CSV = "C:\\Users\\Teert\\AppData\\Roaming\\MetaQuotes\\Terminal\\6BCF14A3917E5BC71FD48812B8ED0586\\MQL4\\Files\\EURUSD,5.csv"
# PATH_TO_LOAD_MODEL = "F:\\dir\\model.h5"
# PATH_TO_MODEL = "F:\\model.h5"

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
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import keras
from keras.layers import PReLU
#import ipdb
from tensorflow.python.keras import layers, initializers, models

import pandas as pd

def make_model(env):
    original_matrix = np.genfromtxt(PATH_TO_CSV, delimiter=",", skip_header=39)
    stateCnt = original_matrix.shape[1]
    #input_shape = (10, stateCnt)
    # dataframe1 = pd.read_csv(PATH_TO_CSV, header=None, usecols=range(0,67), skiprows=39, engine='python')
    #dataframe.assign(s=dataframe.s.shift(-1)).drop(dataframe.index[-1])
    #dataframe[dataframe.columns[-1]] = dataframe[dataframe.columns[-1]].shift(-1)
    # dataframe1 = dataframe1[-1000:]
    # dataset1 = dataframe1.values
    # dataset1 = dataset1.astype('float64')
    #UNCOMMENT THIS TO SCALE ACCORDING TO CURRENT DATA
    # from sklearn import preprocessing
    # scaler1 = preprocessing.MinMaxScaler()
    # X1 = dataset1[:,0:67]
    #scaler1.fit(X1)
    # X1 = scaler1.transform(X1)
    # input_shape = X1.reshape(-1,10,67)
    #input_shape = observation_space
    # input_shape = np.reshape(-1,10,67)
    # num_frames = len(env.observation_space.spaces)
    # height, width = env.observation_space.spaces[0].shape
    # input_shape = height, width, num_frames
    input_shape=(10, stateCnt)
    ph_state = layers.Input(shape=input_shape)
    # conv1 = layers.Conv2D(32, (8, 8), strides=(4, 4))(ph_state)
    conv1 = layers.Conv1D(filters=2, kernel_size=1)(ph_state)
    conv1 = layers.Activation('relu')(conv1)
    # conv2 = layers.Conv2D(64, (4, 4), strides=(2, 2))(conv1)
    # conv2 = layers.Activation('relu')(conv2)
    # conv3 = layers.Conv2D(64, (3, 3), strides=(1, 1))(conv2)
    # conv3 = layers.Activation('relu')(conv3)
    conv_flat = layers.Flatten()(conv1)
    feature = layers.Dense(512)(conv_flat)
    feature = layers.Activation('relu')(feature)
    # actor (policy) and critic (value) streams
    size_logits = 4
    size_value = 4
    logits_init = initializers.RandomNormal(stddev=1e-3)
    logits = layers.Dense(size_logits, kernel_initializer=logits_init)(feature)
    value = layers.Dense(size_value)(feature)
    return models.Model(inputs=ph_state, outputs=[logits, value])

    # modela = Sequential()
    # #print(self.actionCnt)
    # # model.add(Dense(input_shape=(100,stateCnt),filters=32, kernel_size=1, dilation_rate=2, activation='relu',padding='valid', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.005)))
    # modela.add(Conv1D(filters=32,kernel_size=1, dilation_rate=2, activation='relu',padding='valid', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.005), input_shape=(10, stateCnt)))
    # modela.add(Dropout(0.2))
    # modela.add(Conv1D(kernel_size = (1), filters = 32, dilation_rate=4, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modela.add(Dropout(0.2))
    # modela.add(Conv1D(kernel_size = (1), filters = 64, dilation_rate=8, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modela.add(Dropout(0.2))
    # modela.add(Conv1D(kernel_size = (1), filters = 64, dilation_rate=16, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modela.add(MaxPooling1D())
    # modela.add(Dropout(0.2))
    # modela.add(Conv1D(kernel_size = (1), filters = 64, dilation_rate=32, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modela.add(Dropout(0.2))
    # modela.add(Conv1D(kernel_size = (1), filters = 128, strides=2, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modela.add(Dropout(0.2))
    # modela.add(Conv1D(kernel_size = (1), filters = 128, strides=2, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modela.add(Dropout(0.2))
    # modela.add(Conv1D(kernel_size = (1), filters = 128, strides=2, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modela.add(Dropout(0.2))
    # modela.add(Bidirectional(GRU(48,return_sequences=True),merge_mode='concat'))
    # #model.add(Conv1D(filters=100,kernel_size=3,activation='relu',padding='same',input_shape=(10, self.stateCnt)))
    # #model.add(Bidirectional(CuDNNGRU(48,return_sequences=True),merge_mode='concat'))
    # modela.add(PReLU())
    # modela.add(Flatten())
    # modela.add(Dense(24, use_bias=False))
    # #model.add(LeakyReLU(alpha=0.1))
    # modela.add(PReLU())
    # modela.add(Dense(12, use_bias=False))
    # #model.add(LeakyReLU(alpha=0.1))
    # modela.add(PReLU())
    # modela.add(Dense(4, activation="linear"))
    # #adamw =  AdamW(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, weight_decay=0.025, batch_size=1, samples_per_epoch=1, epochs=1)
    # adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # modela.compile(loss="mse" , optimizer="adam" , metrics=["accuracy"])
    #
    # modelb = Sequential()
    # #print(self.actionCnt)
    # # model.add(Dense(input_shape=(100,stateCnt),filters=32, kernel_size=1, dilation_rate=2, activation='relu',padding='valid', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.005)))
    # modelb.add(Conv1D(filters=32,kernel_size=1, dilation_rate=2, activation='relu',padding='valid', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.005), input_shape=(10, stateCnt)))
    # modelb.add(Dropout(0.2))
    # modelb.add(Conv1D(kernel_size = (1), filters = 32, dilation_rate=4, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modelb.add(Dropout(0.2))
    # modelb.add(Conv1D(kernel_size = (1), filters = 64, dilation_rate=8, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modelb.add(Dropout(0.2))
    # modelb.add(Conv1D(kernel_size = (1), filters = 64, dilation_rate=16, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modelb.add(MaxPooling1D())
    # modelb.add(Dropout(0.2))
    # modelb.add(Conv1D(kernel_size = (1), filters = 64, dilation_rate=32, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modelb.add(Dropout(0.2))
    # modelb.add(Conv1D(kernel_size = (1), filters = 128, strides=2, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modelb.add(Dropout(0.2))
    # modelb.add(Conv1D(kernel_size = (1), filters = 128, strides=2, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modelb.add(Dropout(0.2))
    # modelb.add(Conv1D(kernel_size = (1), filters = 128, strides=2, padding='valid', kernel_initializer='he_normal', activation='relu',kernel_regularizer=keras.regularizers.l2(0.005)))
    # modelb.add(Dropout(0.2))
    # modelb.add(Bidirectional(GRU(48,return_sequences=True),merge_mode='concat'))
    # #model.add(Conv1D(filters=100,kernel_size=3,activation='relu',padding='same',input_shape=(10, self.stateCnt)))
    # #model.add(Bidirectional(CuDNNGRU(48,return_sequences=True),merge_mode='concat'))
    # modelb.add(PReLU())
    # modelb.add(Flatten())
    # modelb.add(Dense(24, use_bias=False))
    # #model.add(LeakyReLU(alpha=0.1))
    # modelb.add(PReLU())
    # modelb.add(Dense(12, use_bias=False))
    # #model.add(LeakyReLU(alpha=0.1))
    # modelb.add(PReLU())
    # modelb.add(Dense(4, activation="linear"))
    # #adamw =  AdamW(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, weight_decay=0.025, batch_size=1, samples_per_epoch=1, epochs=1)
    # adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # modelb.compile(loss="mse" , optimizer="adam" , metrics=["accuracy"])
    #
    # conv1 = layersa.Conv2D(32, (8, 8), strides=(4, 4))(ph_state)
    # conv1 = layersa.Activation('relu')(conv1)
    # conv2 = layers.Conv2D(64, (4, 4), strides=(2, 2))(conv1)
    # conv2 = layers.Activation('relu')(conv2)
    # conv3 = layers.Conv2D(64, (3, 3), strides=(1, 1))(conv2)
    # conv3 = layers.Activation('relu')(conv3)
    # conv_flat = layers.Flatten()(conv3)
    #
    # feature = layers.Dense(512)(conv_flat)
    # feature = layers.Activation('relu')(feature)
    # actor (policy) and critic (value) streams
    # logits = model.add(Dense(4, activation="linear"))
    # value = layers.Dense(4)(feature)
    #return models.Model(inputs=ph_state, outputs=[modela, modelb])
    #return model

if __name__ == '__main__':
    trainer = make_trainer(
        algorithm='acer',
        env_maker=lambda: gym.make("EuroDolTrain-v0"),
        model_maker=make_model,
        num_parallel=1,
        train_steps=1000,
        online_learning=False,
        verbose=True,
        batch_size=1,
        save_dir=PATH_TO_MODEL
        )
    trainer.run()

# if __name__ == '__main__':
#     trainer = make_trainer(
#        algorithm='impala',
#        env_maker=lambda: gym.make("EuroDolTrain-v0"),
#        feature_maker=lambda obs_space: make_feature(obs_space, [200, 100]),
#        num_parallel=1,
#        train_steps=1000,
#        online_learning=False,
#        verbose=True,
#        batch_size=100,
#        save_dir="F:\\model.h5"
#        )
#     trainer.run()

    evaluator = make_evaluator(
        env_maker=lambda: gym.make("EuroDolEval-v0"),
        render_timestep=1,
        load_model=PATH_TO_LOAD_MODEL,
        render_end=False,
        num_episode=1,
        algorithm='acer',
        verbose=True,
        )
    evaluator.run()
