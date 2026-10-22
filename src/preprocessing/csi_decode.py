
import numpy as np


def decode_csi(words):
    """Decode 256 BCM4366c0 CSI words with packet-wise scaling."""

    words = np.asarray(words, dtype=np.uint32)

    if words.shape != (256,):
        raise ValueError("Expected 256 CSI words")

    real_mag = ((words >> 18) & 0x7FF).astype(np.int64)
    imag_mag = ((words >> 6) & 0x7FF).astype(np.int64)

    real_sign = (words >> 29) & 1
    imag_sign = (words >> 17) & 1

    exponent = (words & 0x3F).astype(np.int64)
    exponent[exponent >= 32] -= 64

    # Find the maximum exponent-adjusted magnitude
    maxbit = -32

    for i in range(256):
        magnitude = int(real_mag[i] | imag_mag[i])

        if magnitude:
            effective_bit = int(exponent[i]) + magnitude.bit_length() - 1
            maxbit = max(maxbit, effective_bit)

    shift = 10 - maxbit

    real = np.empty(256, dtype=np.float64)
    imag = np.empty(256, dtype=np.float64)

    for i in range(256):
        e = int(exponent[i]) + shift

        for magnitude, sign, output in (
            (real_mag, real_sign, real),
            (imag_mag, imag_sign, imag),
        ):
            value = int(magnitude[i])

            if e < -12:
                value = 0
            elif e < 0:
                value >>= -e
            else:
                value <<= e

            if sign[i]:
                value = -value

            output[i] = value

    return real + 1j * imag


def calculate_amplitude(csi):
    amplitude = np.abs(csi).astype(np.float32)

    if not np.isfinite(amplitude).all():
        raise ValueError("Invalid CSI amplitude")

    return amplitude
