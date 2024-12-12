import os
import tarfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Chargement des fichiers nécessaires
seq_file = "data/proteins/seqs_S60.fa"
classification_file = "data/proteins/domain_classification.txt"
superfamily_file = "data/proteins/superfamily_names.txt"


# Fonction pour lire le fichier FASTA
def read_fasta(file_path):
    sequences = []
    ids = []
    with open(file_path) as f:
        seq = ""
        seq_id = ""
        for line in f:
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    ids.append(seq_id)
                seq_id = line.split("|")[-1].strip()
                seq = ""
            else:
                seq += line.strip()
        if seq:
            sequences.append(seq)
            ids.append(seq_id)
    return pd.DataFrame({"DomainID": ids, "Sequence": sequences})


# Chargement des données de séquence
sequences_df = read_fasta(seq_file)


# Chargement des classifications sans l'en-tête
classifications_df = pd.read_csv(
    classification_file,
    sep="\s+",
    header=None,
    comment="#",
    names=[
        "DomainID",
        "C",
        "A",
        "T",
        "H",
        "S",
        "O",
        "L",
        "I",
        "D",
        "Length",
        "Resolution",
    ],
)


# Chargement des noms de superfamilles
superfamilies_df = pd.read_csv(
    superfamily_file,
    sep="\t",
    header=0,
    names=["data/proteins", "S35_REPS", "TotalDomains", "Description"],
)
# Filtrage pour les superfamilles avec <1000 membres
filtered_superfamilies = superfamilies_df[superfamilies_df["TotalDomains"] < 1000][
    "data/proteins"
][:5]
print(filtered_superfamilies)
# Fusion des données
merged_df = pd.merge(classifications_df, sequences_df, on="DomainID")
merged_df = merged_df[merged_df["H"].isin(filtered_superfamilies)]

# Préparation des données
le = LabelEncoder()
merged_df["Superfamily"] = le.fit_transform(merged_df["H"])

X = merged_df["Sequence"].apply(lambda x: [ord(c) for c in x])
y = merged_df["Superfamily"]

# Encodage des séquences et padding
X_padded = pad_sequences(X, padding="post", maxlen=300)
y_encoded = to_categorical(y)

X_train, X_val, y_train, y_val = train_test_split(
    X_padded, y_encoded, test_size=0.2, random_state=42
)


# Modèle simple avec LSTM
model = Sequential(
    [
        Embedding(input_dim=256, output_dim=128, input_length=300),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(len(le.classes_), activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Early stopping
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Entraînement
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stopping],
)

# Plot des performances
plt.figure(figsize=(12, 6))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Performance du modèle")
plt.show()

# Table d'évaluation
results = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {results[0]:.4f}, Validation Accuracy: {results[1]:.4f}")

from keras_tuner import Hyperband


def build_model(hp):
    model = Sequential(
        [
            Embedding(
                input_dim=256,
                output_dim=hp.Choice("embedding_dim", [64, 128, 256]),
                input_length=300,
            ),
            LSTM(
                hp.Int("lstm_units", min_value=32, max_value=128, step=32),
                return_sequences=False,
            ),
            Dropout(hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)),
            Dense(
                hp.Int("dense_units", min_value=32, max_value=128, step=32),
                activation="relu",
            ),
            Dense(len(le.classes_), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Tuner
tuner = Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=20,
    directory="hyperparam_tuning",
    project_name="protein_superfamily",
)

# Recherche
stop_early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
tuner.search(X_train, y_train, validation_data=(X_val, y_val), callbacks=[stop_early])

# Récupération du meilleur modèle
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
