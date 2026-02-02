### GAN
### By Aliye Hashemi
### Henry Ford Health System --- Hermelin Brain Tumor Center

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# data loading

excel_path = r"C:\NN_lt_049_patients_1-15.xlsx"

X_df = pd.read_excel(excel_path, sheet_name="Input", header=None)
y_df = pd.read_excel(excel_path, sheet_name="Labels", header=None)
p_df = pd.read_excel(excel_path, sheet_name="Patients", header=None)

X_all = X_df.to_numpy()
y_all = y_df.to_numpy().ravel()
patients_all = p_df.to_numpy().ravel()

# patient filtering

gan_patient_ids = np.arange(1, 19)

train_mask = np.isin(patients_all, gan_patient_ids)
X_train = X_all[train_mask]
y_train = y_all[train_mask]

print("Training data:", X_train.shape)
print("Training labels:", y_train.shape)

# normalization

X_min = X_train.min()
X_max = X_train.max()
scale = (X_max - X_min) + 1e-12
X_train_norm = (X_train - X_min) / scale

n_features = X_train_norm.shape[1]
latent_dim = 128

# model definitions

def build_generator(latent_dim, n_features):
    z_in = Input(shape=(latent_dim,))
    y_in = Input(shape=(1,))

    x = layers.Concatenate()([z_in, y_in])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)
    out = layers.Dense(n_features, activation="sigmoid")(x)

    return Model([z_in, y_in], out, name="generator")


def build_discriminator(n_features):
    x_in = Input(shape=(n_features,))
    y_in = Input(shape=(1,))

    x = layers.Concatenate()([x_in, y_in])
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return Model([x_in, y_in], out, name="discriminator")


generator = build_generator(latent_dim, n_features)
discriminator = build_discriminator(n_features)
discriminator.compile(optimizer="adam", loss="binary_crossentropy")

discriminator.trainable = False
z = Input(shape=(latent_dim,))
y = Input(shape=(1,))
fake_x = generator([z, y])
gan_out = discriminator([fake_x, y])
gan = Model([z, y], gan_out)
gan.compile(optimizer="adam", loss="binary_crossentropy")

# training loop

epochs = 5000
batch_size = 32
rng = np.random.default_rng(42)

n_train = X_train_norm.shape[0]

for epoch in range(epochs):
    idx = rng.integers(0, n_train, size=batch_size)
    real_x = X_train_norm[idx]
    real_y = y_train[idx].reshape(-1, 1)

    noise = rng.normal(0, 1, size=(batch_size, latent_dim))
    fake_labels = y_train[rng.integers(0, n_train, size=batch_size)].reshape(-1, 1)
    fake_x = generator.predict([noise, fake_labels], verbose=0)

    x_combined = np.vstack([real_x, fake_x])
    y_combined = np.vstack([
        np.ones((batch_size, 1)),
        np.zeros((batch_size, 1))
    ])
    labels_combined = np.vstack([real_y, fake_labels])

    discriminator.train_on_batch([x_combined, labels_combined], y_combined)

    noise = rng.normal(0, 1, size=(batch_size, latent_dim))
    fake_labels = y_train[rng.integers(0, n_train, size=batch_size)].reshape(-1, 1)
    gan.train_on_batch([noise, fake_labels], np.ones((batch_size, 1)))

    if epoch % 500 == 0:
        print(f"epoch {epoch}")


# synthetic data generation

num_synthetic = 380 # depends on which patients we are training on

synthetic_labels = rng.choice(y_train, size=num_synthetic).reshape(-1, 1)
noise = rng.normal(0, 1, size=(num_synthetic, latent_dim))
synthetic_norm = generator.predict([noise, synthetic_labels], verbose=0)

synthetic_features = synthetic_norm * scale + X_min
synthetic_full = np.hstack([synthetic_features, synthetic_labels])

# save output

out_dir = os.path.dirname(excel_path)
out_path = os.path.join(out_dir, f"Synthetic_Data_{num_synthetic}_with_labels.xlsx")

pd.DataFrame(synthetic_full).to_excel(out_path, index=False, header=False)
print("Saved synthetic data to:", out_path)
