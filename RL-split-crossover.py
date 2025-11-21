import numpy as np
from scipy import signal
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from pydub import AudioSegment


def loading_audio_file(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)               # Convert to mono

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.max(np.abs(samples))

    sample_rate = audio.frame_rate              # Get sample rate

    # Print summary
    print(f"Audio duration: {len(samples)/sample_rate:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")

    return samples, sample_rate

def lowpass_filter(samples, order, cutoff, fs):
    """Stable low-pass Butterworth using second-order sections."""
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, samples)

def highpass_filter(samples, order, cutoff, fs):
    """Stable high-pass Butterworth using second-order sections."""
    sos = signal.butter(order, cutoff, btype='high', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, samples)

def create_stereo_signal(left, right):
    """Create a stereo signal from two mono signals."""
    stereo = np.asarray(np.column_stack((left, right)))
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("stereo_signal must have shape (N, 2)")

    # Normalize to prevent clipping, then convert to int16 PCM
    maxabs = np.max(np.abs(stereo))
    if not np.isfinite(maxabs) or maxabs == 0:
        maxabs = 1.0
    stereo = np.clip(stereo / maxabs, -1.0, 1.0)
    pcm16 = (stereo * 32767.0).astype(np.int16)
    return pcm16

def to_mp3(pcm16_stereo, sample_rate, out_wav='output.wav', out_mp3='output.mp3'):
    """Export stereo PCM16 signal to MP3 file."""

    audio = AudioSegment(
        data=pcm16_stereo.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=2,
    )
    audio.export(out_mp3, format='mp3')

def plot_spectrum(label, x, fs, xlim=None):
    N = len(x)
    window = np.hanning(N)
    X = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(N, 1/fs)
    mag_db = 20 * np.log10(np.maximum(np.abs(X), 1e-12))
    plt.figure(figsize=(10,4))
    plt.plot(freqs, mag_db, label=label)
    if xlim:
        plt.xlim(0, xlim)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title(f'Spectrum: {label}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters (adjust cutoff to e.g. 200-300 Hz to isolate bass)
    order = 6            # Moderate order; too high can cause numerical issues
    cutoff = 2200         # Low-pass cutoff in Hz for "low frequency" content
    audio_path = "workshop-delefilter/06-Californication.mp3"
    output_mp3 = "split-Californication.mp3"

    samples, sample_rate = loading_audio_file(audio_path)
    # print_signal_stats("Original", samples)

    # plot_spectrum("Original", samples, sample_rate, xlim=20000)

    # Apply low-pass filter
    lowFreqComponent = lowpass_filter(samples, order, cutoff, sample_rate)
    highFreqComponent = highpass_filter(samples, order, cutoff, sample_rate)
    plot_spectrum("Low-pass", lowFreqComponent, sample_rate, xlim=20000)
    plot_spectrum("High-pass", highFreqComponent, sample_rate, xlim=20000)

    pcm16_stereo = create_stereo_signal(lowFreqComponent, highFreqComponent)
    to_mp3(pcm16_stereo, sample_rate, out_mp3=output_mp3)

    # Sanity check: difference (high-frequency residue)
    # highResidue = samples - lowFreqComponent
    # print_signal_stats("Residual (high freq)", highResidue)
    # Optional: plot_spectrum("Residual", highResidue, sample_rate, xlim=2000)
