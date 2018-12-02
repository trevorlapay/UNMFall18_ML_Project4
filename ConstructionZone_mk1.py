import scipy
import numpy as np
import keras
import os
import shutil
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from keras.optimizers import SGD, Adam

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
test_dir = 'all/test'
percent_validate = .1

# Move files into respective directories for each class.
# Only do this once (luckily...)
def processFilesForImageDataGenerator():
    df = readTrainingFile()
    for index, row in df.iterrows():
        for filename in os.listdir(green_only_dir):
            if filename.startswith(row['Id']):
                num = row['Target'].split()
                SourceFolder = os.path.join(green_only_dir, filename)
                shutil.copy(SourceFolder, os.path.join(train_dir, num[0]))



def moveTrainFilesToValidationDir():
    for n in range(27):
        numDir = os.path.join(train_dir, str(n))
        for count, file in enumerate(os.listdir(numDir)):
            if count/len(os.listdir(numDir)) < percent_validate:
                newfile = os.path.join(os.path.join(valid_dir, str(n)), file)
                shutil.move(os.path.join(numDir, file), newfile)

def readTrainingFile():
    return pd.read_csv("all/train.csv")

def loadAll():
    train_datagen = ImageDataGenerator()

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            valid_dir,
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
            directory=test_dir,
            target_size=(256, 256),
            color_mode="rgb",
            batch_size = 1,
            class_mode = "categorical",
            shuffle = True,
            seed = 42)


    model = Sequential()
    model.add(Conv2D(32, kernel_size=(4, 4), strides=(1, 1), input_shape=(256, 256, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, kernel_size=(4, 4), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(all_classes)))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    model.fit_generator(
            generator=train_generator,
            steps_per_epoch=100,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=800)
    serialize(model)
    model.evaluate_generator(generator=validation_generator)
    test_generator.reset()
    pred = model.predict_generator(test_generator, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    filenames = test_generator.filenames
    results = pd.DataFrame({"Filename": filenames,
                            "Predictions": predictions})
    results.to_csv("results.csv", index=False)

# Serialize weights.
def serialize(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved to disk")

def main():
    loadAll()

if __name__ == "__main__": main()




