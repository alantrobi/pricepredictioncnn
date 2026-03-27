import numpy as np
import os
import cv2
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

def load_dataset(company):

    X, y = [], []
    folder = f"dataset_img/{company}"

    for file in os.listdir(folder):

        if file.endswith(".png"):

            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)

            label = np.load(os.path.join(folder, file.replace(".png", "_y.npy")))

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)


def build_model(input_shape):

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,1)),

        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train(company):

    print(f"\nTraining: {company}")

    X, y = load_dataset(company)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    model = build_model(X.shape[1:])

    early_stop = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=30,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    loss, mae = model.evaluate(X_test, y_test)

    print(f"{company} MAE (change): {mae:.4f}")

    model.save(f"{company}_img_model.keras")


train("tcs")
train("bajaj")
train("vedanta")