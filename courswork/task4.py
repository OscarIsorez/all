import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras_tuner import RandomSearch


# Step 1: Data Loading
def load_data(file_path):
    # .fa file
    data = pd.read_csv(file_path, delimiter=" ", header=None)

    data.columns = [
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
    ]
    sequences = {}  # Load sequences from a FASTA or similar file
    # Example: Add your logic here to extract sequences from FASTA files.
    return data, sequences


# Step 2: Filter Top Categories
def filter_top_categories(data, sequences, max_members=1000):
    counts = data["H"].value_counts()
    top_categories = counts[counts < max_members].head(5).index
    filtered = data[data["H"].isin(top_categories)]
    filtered["sequence"] = filtered["DomainID"].map(sequences)
    return filtered[["sequence", "H"]].dropna()


# Step 3: Preprocess Data
def preprocess_data(dataset):
    # Tokenize sequences
    alphabet = set("".join(dataset["sequence"]))
    vocab = {char: idx for idx, char in enumerate(alphabet, start=1)}
    dataset["encoded_seq"] = dataset["sequence"].apply(
        lambda x: [vocab[char] for char in x]
    )

    # Pad sequences
    max_length = max(dataset["encoded_seq"].apply(len))
    dataset["padded_seq"] = pad_sequences(
        dataset["encoded_seq"], maxlen=max_length, padding="post"
    ).tolist()

    # Encode labels
    label_encoder = LabelEncoder()
    dataset["encoded_label"] = label_encoder.fit_transform(dataset["H"])

    return (
        dataset,
        max_length,
        len(vocab) + 1,
        len(label_encoder.classes_),
        label_encoder,
    )


# Step 4: Build Model
def build_model(vocab_size, seq_length, num_classes, dropout_rate=0.2):
    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=64, input_length=seq_length),
            Conv1D(64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Dropout(dropout_rate),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(dropout_rate),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Step 5: Train and Evaluate Model
def train_and_evaluate(data, max_length, vocab_size, num_classes, label_encoder):
    X = np.array(data["padded_seq"].tolist())
    y = np.array(data["encoded_label"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model(vocab_size, max_length, num_classes)
    history = model.fit(
        X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=1
    )

    # Plot Training History
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.show()

    # Evaluate on Test Set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")


# Step 6: Hyperparameter Optimization
def hyperparameter_tuning(data, max_length, vocab_size, num_classes):
    def model_builder(hp):
        model = Sequential(
            [
                Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
                Conv1D(
                    hp.Int("units", min_value=32, max_value=128, step=32),
                    kernel_size=3,
                    activation="relu",
                ),
                MaxPooling1D(pool_size=2),
                Dropout(
                    hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)
                ),
                Flatten(),
                Dense(
                    hp.Int("dense_units", min_value=32, max_value=128, step=32),
                    activation="relu",
                ),
                Dropout(
                    hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)
                ),
                Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    tuner = RandomSearch(
        model_builder,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=1,
        directory="hyperparameter_tuning",
        project_name="protein_function",
    )

    X = np.array(data["padded_seq"].tolist())
    y = np.array(data["encoded_label"])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    best_hps = tuner.get_best_hyperparameters()[0]
    print(f"Best Hyperparameters: {best_hps.values}")


# Step 7: Run
file_path = "data/proteins/seqs_S60.fa"
data, sequences = load_data(file_path)
filtered_data = filter_top_categories(data, sequences)
processed_data, max_length, vocab_size, num_classes, label_encoder = preprocess_data(
    filtered_data
)
train_and_evaluate(processed_data, max_length, vocab_size, num_classes, label_encoder)
hyperparameter_tuning(processed_data, max_length, vocab_size, num_classes)
