import scipy
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

all_classes = ["Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center", "Nuclear speckles",
               "Nuclear bodies", "Endoplasmic reticulum",  "Golgi apparatus",
               "Peroxisomes", "Endosomes", "Lysosomes", "Intermediate filaments", "Actin filaments",
                "Focal adhesion sites", "Microtubules", "Microtubule ends", "Cytokinetic bridge",
                "Mitotic spindle", "Microtubule organizing center", "Cenrosome", "Lipid droplets",
                "Plasma Membrane", "Cell junctions", "Mitochondra", "Aggresome", "Cytosol", "Cytoplasmic Bodies",
                "Rods & Rings"]

def loadAll():
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'all/train',
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            'all/test',
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical')

    model = Sequential()
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(len(all_classes), activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])
    model.fit_generator(
            generator=train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)


def main():
    loadAll()


if __name__ == "__main__": main()




