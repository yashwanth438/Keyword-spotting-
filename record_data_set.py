import serial
import struct
import numpy as np
import librosa

from collections import deque

# =====================================================
# SETTINGS
# =====================================================

PORT = "/dev/ttyACM1"
BAUD = 921600

START_MARKER = b'\xAA\xBB\xCC\xDD'

MFCC_DIM = 40
FRAME_BYTES = MFCC_DIM * 4

TIME_FRAMES = 64

# -----------------------------------------------------
# CHANGE THIS BEFORE RECORDING
# -----------------------------------------------------

LABEL_NAME = "unknown"

# =====================================================
# UART
# =====================================================

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

print("UART connected")

# =====================================================
# BUFFERS
# =====================================================

mfcc_buffer = deque(
    maxlen=TIME_FRAMES
)

dataset = []

# =====================================================
# MAIN LOOP
# =====================================================

print()
print("Speak the word repeatedly...")
print("Press Ctrl+C when done")
print()

try:

    while True:

        # -------------------------------------------------
        # FIND START MARKER
        # -------------------------------------------------

        first = ser.read(1)

        if first != b'\xAA':
            continue

        marker = first + ser.read(3)

        if marker != START_MARKER:
            continue

        # -------------------------------------------------
        # READ FRAME
        # -------------------------------------------------

        data = b''

        while len(data) < FRAME_BYTES:

            chunk = ser.read(
                FRAME_BYTES - len(data)
            )

            if len(chunk) == 0:
                break

            data += chunk

        if len(data) != FRAME_BYTES:
            continue

        # -------------------------------------------------
        # UNPACK MFCC
        # -------------------------------------------------

        mfcc_frame = np.array(
            struct.unpack(
                "<40f",
                data
            ),
            dtype=np.float32
        )

        if not np.all(
            np.isfinite(mfcc_frame)
        ):
            continue

        mfcc_buffer.append(
            mfcc_frame
        )

        if len(mfcc_buffer) < TIME_FRAMES:
            continue

        # -------------------------------------------------
        # BUILD MFCC MATRIX
        # -------------------------------------------------

        mfcc = np.array(
            mfcc_buffer,
            dtype=np.float32
        ).T

        # -------------------------------------------------
        # DELTA
        # -------------------------------------------------

        delta = librosa.feature.delta(
            mfcc,
            width=9
        )

        # -------------------------------------------------
        # DELTA2
        # -------------------------------------------------

        delta2 = librosa.feature.delta(
            mfcc,
            order=2,
            width=9
        )

        # -------------------------------------------------
        # CONCAT
        # -------------------------------------------------

        features = np.concatenate(
            [
                mfcc,
                delta,
                delta2
            ],
            axis=0
        )

        features = features.T

        if not np.all(
            np.isfinite(features)
        ):
            continue

        dataset.append(
            features.astype(
                np.float32
            )
        )

        print(
            f"\rSamples collected: "
            f"{len(dataset)}",
            end=""
        )

except KeyboardInterrupt:

    print()

# =====================================================
# SAVE
# =====================================================

dataset = np.array(
    dataset,
    dtype=np.float32
)

filename = f"{LABEL_NAME}.npy"

np.save(
    filename,
    dataset
)

print()
print(
    f"Saved {len(dataset)} samples to "
    f"{filename}"
)

ser.close()