import numpy as np
import os
from PIL import Image
import imgaug as ia
import pandas as pd
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, ReLU, LeakyReLU
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import f1_score as off1
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from keras.optimizers import SGD, Adam, Nadam
from imgaug import augmenters as iaa
import cv2
from keras.backend.tensorflow_backend import set_session
all_classes = ["Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center", "Nuclear speckles",
               "Nuclear bodies", "Endoplasmic reticulum",  "Golgi apparatus",
               "Peroxisomes", "Endosomes", "Lysosomes", "Intermediate filaments", "Actin filaments",
                "Focal adhesion sites", "Microtubules", "Microtubule ends", "Cytokinetic bridge",
                "Mitotic spindle", "Microtubule organizing center", "Cenrosome", "Lipid droplets",
                "Plasma Membrane", "Cell junctions", "Mitochondra", "Aggresome", "Cytosol", "Cytoplasmic Bodies",
                "Rods & Rings"]

green_only_dir='all/train/green_only'
train_dir = 'all/train'
valid_dir = 'all/valid'
test_dir = 'all/test/test'
sample_dir = 'all/sample_submission.csv'
epochs = 100
BATCH_SIZE = 128
SEED = 777
SHAPE = (192, 192, 4)
DIR = ''
VAL_RATIO = 0.1 # 10 % as validation
THRESHOLD = 0.05 # due to different cost of True Positive vs False Positive, this is the probability threshold to predict the class as 'yes'

ia.seed(SEED)

def readTrainingFile():
    return pd.read_csv("all/train.csv")


def getTrainDataset():

    data = pd.read_csv(DIR + 'all/train.csv')

    paths = []
    labels = []

    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(train_dir, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


def getTestDataset():
    data = pd.read_csv(sample_dir)

    paths = []
    labels = []

    for name in data['Id']:
        y = np.ones(28)
        paths.append(os.path.join(test_dir, name))
        labels.append(y)

    return np.array(paths), np.array(labels)

def generateSubmitFile(model=None):
    pathsTest, labelsTest = getTestDataset()
    if not model:
        model = load_model('./base.model', custom_objects={'f1': f1})  # , 'f1_loss': f1_loss})
    paths, labels = getTrainDataset()
    testg = ProteinDataGenerator(pathsTest, labelsTest, BATCH_SIZE, SHAPE)
    lastTrainIndex = int((1 - VAL_RATIO) * paths.shape[0])
    pathsVal = paths[lastTrainIndex:]
    labelsVal = labels[lastTrainIndex:]
    vg = ProteinDataGenerator(pathsVal, labelsVal, BATCH_SIZE, SHAPE, use_cache=True, shuffle=False)
    submit = pd.read_csv(sample_dir)
    P = np.zeros((pathsTest.shape[0], 28))
    for i in tqdm(range(len(testg))):
        images, labels = testg[i]
        score = model.predict(images)
        P[i * BATCH_SIZE:i * BATCH_SIZE + score.shape[0]] = score
    rng = np.arange(0, 1, 0.001)
    f1s = np.zeros((rng.shape[0], 28))
    lastFullValPred = np.empty((0, 28))
    np.random.seed(SEED)
    keys = np.arange(paths.shape[0], dtype=np.int)
    np.random.shuffle(keys)

    fullValGen = vg
    lastFullValLabels = np.empty((0, 28))
    for i in tqdm(range(len(fullValGen))):
        im, lbl = fullValGen[i]
        scores = model.predict(im)
        lastFullValPred = np.append(lastFullValPred, scores, axis=0)
        lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)
    print(lastFullValPred.shape, lastFullValLabels.shape)
    for j, t in enumerate(tqdm(rng)):
        for i in range(28):
            p = np.array(lastFullValPred[:, i] > t, dtype=np.int8)
            scoref1 = off1(lastFullValLabels[:, i], p, average='binary')
            f1s[j, i] = scoref1
    PP = np.array(P)
    prediction = []
    T = np.empty(28)
    for i in range(28):
        T[i] = rng[np.where(f1s[:, i] == np.max(f1s[:, i]))[0][0]]
    print('Probability threshold maximizing CV F1-score for each class:')
    print(T)
    for row in tqdm(range(submit.shape[0])):

        str_label = ''

        for col in range(PP.shape[1]):
            if (PP[row, col] < T[col]):
                str_label += ''
            else:
                str_label += str(col) + ' '
        prediction.append(str_label.strip())

    submit['Predicted'] = np.array(prediction)
    submit.to_csv('submit.csv', index=False)



def fitSerializeModel():
    model = create_model(SHAPE)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Nadam(1e-03),
        metrics=['acc', f1])

    model.summary()

    paths, labels = getTrainDataset()
    keys = np.arange(paths.shape[0], dtype=np.int)
    np.random.seed(SEED)
    np.random.shuffle(keys)
    lastTrainIndex = int((1 - VAL_RATIO) * paths.shape[0])

    pathsTrain = paths[0:lastTrainIndex]
    labelsTrain = labels[0:lastTrainIndex]
    pathsVal = paths[lastTrainIndex:]
    labelsVal = labels[lastTrainIndex:]

    print(paths.shape, labels.shape)
    print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)

    tg = ProteinDataGenerator(pathsTrain, labelsTrain, BATCH_SIZE, SHAPE, use_cache=True, augment=False2, shuffle=True)
    vg = ProteinDataGenerator(pathsVal, labelsVal, BATCH_SIZE, SHAPE, use_cache=True, shuffle=True)

    # https://keras.io/callbacks/#modelcheckpoint
    checkpoint = ModelCheckpoint('./base.model', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)
    reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')

    use_multiprocessing = False  # DO NOT COMBINE MULTIPROCESSING WITH CACHE!
    workers = 1  # DO NOT COMBINE MULTIPROCESSING WITH CACHE!

    hist = model.fit_generator(
        tg,
        steps_per_epoch=len(tg),
        validation_data=vg,
        validation_steps=8,
        epochs=epochs,
        use_multiprocessing=use_multiprocessing,
        workers=workers,
        verbose=1,
        callbacks=[checkpoint])

    serialize(model)

def f1(y_true, y_pred):
    # y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def create_model(input_shape):
    dropRate = 0.30

    init = Input(input_shape)
    x = BatchNormalization(axis=-1)(init)
    x = Conv2D(8, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(8, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(16, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    c1 = Conv2D(16, (3, 3), padding='same')(x)
    c1 = ReLU()(c1)
    c2 = Conv2D(16, (5, 5), padding='same')(x)
    c2 = ReLU()(c2)
    c3 = Conv2D(16, (7, 7), padding='same')(x)
    c3 = ReLU()(c3)
    c4 = Conv2D(16, (1, 1), padding='same')(x)
    c4 = ReLU()(c4)
    x = Concatenate()([c1, c2, c3, c4])
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(32, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(64, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Conv2D(128, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropRate)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(28)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.1)(x)
    x = Dense(28)(x)
    x = Activation('sigmoid')(x)

    model = Model(init, x)

    return model

def f1_loss(y_true, y_pred):
    # y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

# Serialize weights.
def serialize(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved to disk")

# load model from JSON file
def loadModelFromJSON():
    model = create_model(SHAPE)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Nadam(1e-03),
        metrics=['acc', f1])
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    return model

class ProteinDataGenerator(keras.utils.Sequence):

    def __init__(self, paths, labels, batch_size, shape, shuffle=False, use_cache=False, augment=False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]

        if self.augment == True:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5),  # horizontal flips
                    # Small gaussian blur with random sigma between 0 and 0.5.
                    # But we only blur about 50% of all images.

                ])], random_order=True)

            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)
            y = np.concatenate((y, y, y, y), 0)

        return X, y

    def on_epoch_end(self):

        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __load_image(self, path):
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        Y = Image.open(path + '_yellow.png')

        im = np.stack((
            np.array(R),
            np.array(G),
            np.array(B),
            np.array(Y)), -1)

        im = cv2.resize(im, (SHAPE[0], SHAPE[1]))
        im = np.divide(im, 255)
        return im

# If using gpu, this likely needs to be run to manage GPU
# when predicting
def setSession():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)

def loadAll():
    # As an alternate to using base.model, you can serialize your model and pass it in to generateSubmit.
    # model = loadModelFromJSON()
    setSession()
    fitSerializeModel()
    generateSubmitFile()

def main():
    loadAll()

if __name__ == "__main__": main()