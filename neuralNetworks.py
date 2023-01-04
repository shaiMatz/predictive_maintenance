import pandas as pd
import tensorflow as tf
from keras.layers import CategoryEncoding, Normalization
from tensorflow import keras
from tensorflow.keras import layers

print("The relevant libraries have been imported.")

# Load the dataset
dataframe = pd.read_csv("mill.csv")

# Split the data into training and validation sets
val_dataframe = dataframe.sample(frac=0.01, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(f"Using {len(train_dataframe)} samples for training and {len(val_dataframe)} for validation")


# Convert the dataframes to datasets
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("VB")
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    dataset = dataset.shuffle(buffer_size=len(dataframe))
    return dataset


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

# Print a sample from the training dataset
for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

# Batch the datasets
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)


# Encode numerical and categorical features
def encode_numerical_feature(feature, name, dataset):
    normalizer = Normalization()
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    normalizer.adapt(feature_ds)
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_integer_categorical_feature(feature, name, dataset):
    encoder = CategoryEncoding(num_tokens=5, output_mode="binary")
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    encoder.adapt(feature_ds)
    encoded_feature = encoder(feature)
    return encoded_feature


# Integer categorical features
DOC = keras.Input(shape=(1,), name="DOC", dtype="int64")
feed = keras.Input(shape=(1,), name="feed", dtype="int64")
material = keras.Input(shape=(1,), name="material", dtype="int64")

# Continuous features
smcDC = keras.Input(shape=(1,), name="smcDC")
time = keras.Input(shape=(1,), name="time")
vib_spindle = keras.Input(shape=(1,), name="vib_spindle")

# Encode features
DOC_encoded = encode_integer_categorical_feature(DOC, "DOC", train_ds)
feed_encoded = encode_integer_categorical_feature(feed, "feed", train_ds)
material_encoded = encode_integer_categorical_feature(material, "material", train_ds)
smcDC_encoded = encode_numerical_feature(smcDC, "smcDC", train_ds)
time_encoded = encode_numerical_feature(time, "time", train_ds)
vib_spindle_encoded = encode_numerical_feature(vib_spindle, "vib_spindle", train_ds)

# Concatenate encoded features
features = layers.concatenate(
    [DOC_encoded, feed_encoded, material_encoded, smcDC_encoded, time_encoded, vib_spindle_encoded])

# Define model architecture
x = layers.Dense(64, activation="relu")(features)
x = layers.Dense(64, activation="relu")(x)
predictions = layers.Dense(1, activation="sigmoid")(x)

# Compile the model
model = keras.Model(inputs=[DOC, feed, material, smcDC, time, vib_spindle], outputs=predictions)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(train_ds, epochs=5, validation_data=val_ds)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")
