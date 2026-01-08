import os
import numpy as np
import torch
import librosa
import librosa.display
import torchaudio
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

# -----------------------------
# Recordings
# -----------------------------
AUDIO_PATH = "./recorded_audio.wav"

RECORD_SECONDS = 5        # seconds
RECORD_SR = 48000         # original recording SR (keep high)
CHANNELS = 1              # mono

sd.default.device = 16


print("[Recording] Start...")
audio = sd.rec(
    int(RECORD_SECONDS * RECORD_SR),
    samplerate=RECORD_SR,
    channels=CHANNELS,
    dtype="float32"
)
sd.wait()
print("[Recording] Done.")

# Save wav
sf.write(AUDIO_PATH, audio.squeeze(), RECORD_SR)
print(f"[Saved] {AUDIO_PATH}")


# -----------------------------
# Configuration
# -----------------------------

TARGET_SR = 16000
N_MELS = 64
N_FFT = int(TARGET_SR * 0.025)   # 25 ms window
HOP_LENGTH = int(TARGET_SR * 0.01)  # 10 ms hop
OUT_PATH = "./audio_log_melspectrogram.png"

# -----------------------------
# Load audio (keep original SR)
# -----------------------------
waveform, orig_sr = librosa.load(AUDIO_PATH, sr=None, mono=True)

# Convert to torch tensor
waveform_t = torch.from_numpy(waveform).float()

# -----------------------------
# Resample
# -----------------------------
if orig_sr != TARGET_SR:
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=TARGET_SR
    )
    waveform_t = resampler(waveform_t)

waveform_np = waveform_t.numpy()

# -----------------------------
# Mel Spectrogram
# -----------------------------
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0
)

mel_spec = mel_transform(waveform_t)
mel_spec_db = librosa.power_to_db(
    mel_spec.numpy(),
    ref=np.max
)

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(10, 6))

# Waveform
plt.subplot(2, 1, 1)
plt.plot(waveform_np, color="#ed5c9b", linewidth=1)
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.axis("off")

# Mel Spectrogram
plt.subplot(2, 1, 2)
librosa.display.specshow(
    mel_spec_db,
    sr=TARGET_SR,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="mel",
    cmap="magma"
)
plt.title("Log-Mel Spectrogram")
plt.colorbar(format="%+2.0f dB")

plt.tight_layout()
plt.savefig(OUT_PATH)
plt.close()

print(f"[Saved] {OUT_PATH}")
