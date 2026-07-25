import os
import wave
import numpy as np


SAMPLE_RATE = 44100


def envelope(x, decay):
    return np.exp(-x * decay)


def noise(length, amp):
    return np.random.randn(length) * amp


def main():

    duration = 0.12

    t = np.linspace(
        0,
        duration,
        int(SAMPLE_RATE * duration),
        endpoint=False
    )

    # Mechanical click
    click = (
        np.sin(2 * np.pi * 1900 * t) * envelope(t, 120)
    )

    # Sharp transient
    transient = (
        np.sin(2 * np.pi * 4300 * t)
        * envelope(t, 250)
        * 0.45
    )

    # Tiny body resonance
    body = (
        np.sin(2 * np.pi * 620 * t)
        * envelope(t, 35)
        * 0.18
    )

    # Small amount of mechanical noise
    hiss = noise(len(t), 0.015) * envelope(t, 180)

    sound = (
        click +
        transient +
        body +
        hiss
    )

    sound /= np.max(np.abs(sound))
    sound *= 0.95

    pcm = (sound * 32767).astype(np.int16)

    os.makedirs("assets/sounds", exist_ok=True)

    path = "assets/sounds/shutter.wav"

    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())

    print("Created:", os.path.abspath(path))


if __name__ == "__main__":
    main()