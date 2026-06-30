import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading datasets...")

on = np.load("on.npy")
off = np.load("off.npy")
up = np.load("up.npy")
down = np.load("down.npy")
unknown = np.load("unknown.npy")
silence = np.load("silence.npy")

print("on      :", on.shape)
print("off     :", off.shape)
print("up      :", up.shape)
print("down    :", down.shape)
print("unknown :", unknown.shape)
print("silence :", silence.shape)

# =====================================================
# LABELS
# =====================================================

LABELS = [
    "down",     # 0
    "off",      # 1
    "on",       # 2
    "up",       # 3
    "unknown",  # 4
    "silence"   # 5
]

# =====================================================
# BUILD X/Y
# =====================================================

X = np.concatenate([
    down,
    off,
    on,
    up,
    unknown,
    silence
])

Y = np.concatenate([

    np.full(
        len(down),
        0,
        dtype=np.int32
    ),

    np.full(
        len(off),
        1,
        dtype=np.int32
    ),

    np.full(
        len(on),
        2,
        dtype=np.int32
    ),

    np.full(
        len(up),
        3,
        dtype=np.int32
    ),

    np.full(
        len(unknown),
        4,
        dtype=np.int32
    ),

    np.full(
        len(silence),
        5,
        dtype=np.int32
    )
])

print("\nDataset Shape:")
print("X =", X.shape)
print("Y =", Y.shape)

# =====================================================
# SHUFFLE
# =====================================================

idx = np.random.permutation(
    len(X)
)

X = X[idx]
Y = Y[idx]

# =====================================================
# CNN INPUT
# =====================================================

X = X[..., np.newaxis]

print(
    "\nCNN Input Shape:",
    X.shape
)

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    Y,

    test_size=0.20,

    random_state=42,

    stratify=Y
)

print(
    "\nTrain:",
    X_train.shape
)

print(
    "Test :",
    X_test.shape
)

# =====================================================
# MODEL
# =====================================================

model = models.Sequential([

    layers.Input(
        shape=(64, 120, 1)
    ),

    # -------------------------------------------------
    # BLOCK 1
    # -------------------------------------------------

    layers.Conv2D(
        16,
        (3,3),
        activation='relu',
        padding='same'
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2,2)
    ),

    # -------------------------------------------------
    # BLOCK 2
    # -------------------------------------------------

    layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2,2)
    ),

    # -------------------------------------------------
    # BLOCK 3
    # -------------------------------------------------

    layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2,2)
    ),

    # -------------------------------------------------
    # DENSE
    # -------------------------------------------------

    layers.Flatten(),

    layers.Dense(
        128,
        activation='relu'
    ),

    layers.Dropout(
        0.3
    ),

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------

    layers.Dense(
        len(LABELS),
        activation='softmax'
    )
])

# =====================================================
# COMPILE
# =====================================================

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']
)

# =====================================================
# SUMMARY
# =====================================================

model.summary()

# =====================================================
# TRAIN
# =====================================================

history = model.fit(

    X_train,
    y_train,

    epochs=30,

    batch_size=32,

    validation_data=(
        X_test,
        y_test
    )
)

# =====================================================
# EVALUATE
# =====================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(
    f"\nTest Accuracy: "
    f"{accuracy*100:.2f}%"
)

# =====================================================
# SAVE H5
# =====================================================

model.save(
    "kws_model_firmware.h5"
)

print(
    "\nSaved:"
    " kws_model_firmware.h5"
)

# =====================================================
# EXPORT TFLITE
# =====================================================

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

tflite_model = converter.convert()

with open(
    "kws_model_firmware.tflite",
    "wb"
) as f:

    f.write(
        tflite_model
    )

print(
    "Saved:"
    " kws_model_firmware.tflite"
)