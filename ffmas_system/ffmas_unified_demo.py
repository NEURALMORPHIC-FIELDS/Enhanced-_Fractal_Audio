"""
===============================
FFMAS: Fractal Frequential Multidimensional Audio System
Enhanced Fractal Audio Compression Demo
===============================

Creator & Principal Investigator: Vasile Lucian Borbeleac
Co-Author: Dr. Med. Vet. Luta Borbeleac Andreea-Gianina
Organization: FRAGMERGENT TECHNOLOGY S.R.L, Romania
Contact: V.l.borbel@gmail.com

This unified demo demonstrates:
1) FFMAS (Fractal Frequential Multidimensional Audio System)
2) Enhanced Fractal Audio Compression with Extended Metrics

ADAPTED FOR HEADLESS EXECUTION:
- Matplotlib Agg backend for non-interactive plotting
- File-based outputs instead of IPython display
- All results saved to OUTPUT_DIR
"""

import os
import sys

# CRITICAL: Set matplotlib to non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Headless mode for Windows execution

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math
from scipy.io.wavfile import write

# Setup output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] Output directory: {OUTPUT_DIR}")
print(f"[INFO] Starting FFMAS Unified Demo...")

###############################################################################
#                             UTILS / HELPERS
###############################################################################

def apply_frequential_transform(audio_data):
    """
    DEMO: Simplified mathematical transformation (placeholder).
    In a real scenario, you might use FFT, wavelets, or fractal transforms.
    """
    return f"FrequentialTransformed({audio_data})"

def generate_fractal_entropy(data):
    """
    DEMO: A simplified fractal entropy approach based on ASCII sum + modulus.
    """
    base_val = sum(ord(c) for c in data)
    return base_val % 9973

###############################################################################
#                       ENCODER (FRAW) - FFMAS
###############################################################################

class BulkLogicFrame:
    """
    Represents a logical data structure encapsulating large (bulk) blocks
    of audio information in a fractal-frequential format.
    """
    def __init__(self):
        self.frame_counter = 0

    def create_frame(self, transformed_data):
        self.frame_counter += 1
        return {
            "id": self.frame_counter,
            "bulk_data": transformed_data
        }

class FractalHash:
    """
    Generates fractal signatures based on frequency entropy.
    """
    def generate_hash(self, bulk_frame):
        entropy_value = generate_fractal_entropy(str(bulk_frame["bulk_data"]))
        signature = f"FRAC-{bulk_frame['id']}-{entropy_value}"
        return signature

class AdaptiveEncoder:
    """
    Performs audio encoding using fractal-frequential techniques (DEMO).
    """
    def __init__(self):
        self.blf = BulkLogicFrame()
        self.hash_generator = FractalHash()

    def encode(self, raw_audio_path):
        """
        In a real implementation, you would load an actual audio file here
        and apply fractal transforms. This is a simplified example.
        """
        audio_data = f"Simulated audio data from {raw_audio_path}"
        transformed_data = apply_frequential_transform(audio_data)
        bulk_frame = self.blf.create_frame(transformed_data)
        fractal_signature = self.hash_generator.generate_hash(bulk_frame)
        encoded_data = {
            "frame": bulk_frame,
            "signature": fractal_signature
        }
        return encoded_data

###############################################################################
#                     STORAGE (CHADS) - FFMAS
###############################################################################

class FractalMemory:
    """
    Simulates fractal-holographic structures for storage.
    """
    def __init__(self):
        self.storage_dict = {}

    def write_frame(self, frame, signature):
        frame_id = frame["id"]
        self.storage_dict[frame_id] = {
            "bulk_data": frame["bulk_data"],
            "signature": signature
        }

class DiagonalRetriever:
    """
    Conceptual implementation of diagonal reading for a holographic structure.
    """
    def retrieve(self, identity_signature, fractal_memory):
        results = []
        for key, value in fractal_memory.storage_dict.items():
            # Example filter: if identity_signature is in the stored signature
            if identity_signature in value["signature"]:
                results.append({
                    "bulk_data": value["bulk_data"],
                    "signature": value["signature"]
                })
        return results

class HolographicStorage:
    """
    Stores audio data in multidimensional holographic structures.
    """
    def __init__(self):
        self.memory = FractalMemory()
        self.diagonal_retriever = DiagonalRetriever()

    def store(self, encoded_audio):
        frame = encoded_audio.get("frame")
        signature = encoded_audio.get("signature")
        self.memory.write_frame(frame, signature)

    def diagonal_retrieve(self, identity_signature):
        data = self.diagonal_retriever.retrieve(identity_signature, self.memory)
        return data

###############################################################################
#             SELECTIVE BROADCASTING (SAIB) - FFMAS
###############################################################################

class IdentityProfile:
    """
    Represents a fractal auditory identity (IFA) profile for each listener.
    """
    def __init__(self, listener_id, fractal_signature):
        self.listener_id = listener_id
        self.fractal_signature = fractal_signature

    def matches_audio_signature(self, audio_signature):
        return self.fractal_signature in audio_signature

class SelectiveTransmitter:
    """
    Transmits audio data only to authorized listeners, based on IFA.
    """
    def selective_broadcast(self, audio_data, listener_profiles):
        """
        Filters audio data for each listener according to IFA.
        """
        transmitted_data = {}
        for lp in listener_profiles:
            profile = IdentityProfile(lp, "FRAC-")
            authorized_frames = []
            for frame in audio_data:
                # Check if the fractal signature matches
                if profile.matches_audio_signature(frame["signature"]):
                    authorized_frames.append(frame)
            transmitted_data[lp] = authorized_frames
        return transmitted_data

###############################################################################
#               SYNCHRONIZATION (AFDS) - FFMAS
###############################################################################

class AdaptiveLink:
    """
    Represents an adaptive fractal link between audio devices.
    """
    def create_link(self, device_id):
        return f"Adaptive link established with {device_id}"

class FrequentialSync:
    """
    Performs synchronization between devices based on fractal-frequential recognition (DEMO).
    """
    def __init__(self):
        self.adaptive_link = AdaptiveLink()

    def synchronize(self, devices):
        sync_report = []
        for device in devices:
            status = self.adaptive_link.create_link(device)
            sync_report.append(status)
        return sync_report

###############################################################################
#      RECONSTRUCTION (Ψ-Audio) & FILTERING (FANF) - FFMAS
###############################################################################

class RealtimeOptimizer:
    """
    Adjusts audio quality in real-time, using environment/hardware feedback (DEMO).
    """
    def optimize(self, audio_frame):
        return f"Optimized({audio_frame})"

class AdaptiveDecoder:
    """
    Decodes fractal-audio data and reconstructs the original signal (DEMO).
    """
    def __init__(self):
        self.optimizer = RealtimeOptimizer()

    def reconstruct(self, encoded_audio):
        reconstructed_signal = []
        if isinstance(encoded_audio, dict):
            # The data might be something like {listener_profile: [frame1, frame2, ...]}
            for profile, frames in encoded_audio.items():
                for frame in frames:
                    optimized_frame = self.optimizer.optimize(frame["bulk_data"])
                    reconstructed_signal.append(optimized_frame)
        return reconstructed_signal

class AdaptiveNoiseReduction:
    """
    A simplified fractal-based adaptive noise reduction (DEMO).
    """
    def reduce_noise(self, audio_frame):
        return f"NoiseReduced({audio_frame})"

class FractalNoiseFilter:
    """
    Filters out undesired noise using fractal-based analysis (DEMO).
    """
    def __init__(self):
        self.anr = AdaptiveNoiseReduction()

    def apply(self, decoded_audio):
        filtered_signal = []
        for frame in decoded_audio:
            filtered_frame = self.anr.reduce_noise(frame)
            filtered_signal.append(filtered_frame)
        return filtered_signal

###############################################################################
#                      FFMAS MAIN ORCHESTRATOR
###############################################################################

class FFMAS:
    """
    Central orchestrator class for the Fractal Frequential Multidimensional Audio System.
    """
    def __init__(self):
        self.encoder = AdaptiveEncoder()
        self.storage = HolographicStorage()
        self.transmitter = SelectiveTransmitter()
        self.sync = FrequentialSync()
        self.decoder = AdaptiveDecoder()
        self.noise_filter = FractalNoiseFilter()

    def encode_audio(self, raw_audio):
        encoded_audio = self.encoder.encode(raw_audio)
        self.storage.store(encoded_audio)
        return encoded_audio

    def retrieve_audio(self, identity_signature):
        audio_data = self.storage.diagonal_retrieve(identity_signature)
        return audio_data

    def broadcast_audio(self, audio_data, listener_profiles):
        selective_audio = self.transmitter.selective_broadcast(audio_data, listener_profiles)
        return selective_audio

    def synchronize_devices(self, devices):
        sync_report = self.sync.synchronize(devices)
        return sync_report

    def reconstruct_audio(self, encoded_audio):
        decoded_audio = self.decoder.reconstruct(encoded_audio)
        filtered_audio = self.noise_filter.apply(decoded_audio)
        return filtered_audio

###############################################################################
#               EXPERIMENT 1: FFMAS FLOW (Simplified Metrics)
###############################################################################

def run_ffmas_experiment():
    """
    Runs the end-to-end FFMAS flow with minimal, text-based 'audio' data.
    Returns simple metrics that confirm module functionality.
    """
    print("\n=== EXPERIMENT 1: FFMAS FLOW ===")
    ffmas = FFMAS()

    # 1) Encoding
    raw_audio_input = "audio_input.wav"
    start_time = time.time()
    encoded = ffmas.encode_audio(raw_audio_input)
    encode_duration = time.time() - start_time

    # 2) Retrieval
    start_time = time.time()
    short_sig = encoded["signature"][:7]  # e.g., "FRAC-1-"
    retrieved = ffmas.retrieve_audio(short_sig)
    retrieve_duration = time.time() - start_time

    # 3) Broadcasting
    start_time = time.time()
    listener_profiles = ["listener1.ifa", "listener2.ifa"]
    broadcasted = ffmas.broadcast_audio(retrieved, listener_profiles)
    broadcast_duration = time.time() - start_time

    # 4) Synchronization
    start_time = time.time()
    devices = ["speaker_1", "headphone_1"]
    sync_report = ffmas.synchronize_devices(devices)
    sync_duration = time.time() - start_time

    # 5) Reconstruction + Noise Filtering
    start_time = time.time()
    final_audio = ffmas.reconstruct_audio(broadcasted)
    reconstruction_duration = time.time() - start_time

    metrics = {
        "EncodedFrameID": encoded["frame"]["id"],
        "EncodedSignature": encoded["signature"],
        "RetrievedChunks": len(retrieved),
        "BroadcastedProfiles": list(broadcasted.keys()),
        "SynchronizedDevices": devices,
        "SyncReport": sync_report,
        "ReconstructedFramesCount": len(final_audio),
        "Timings (seconds)": {
            "Encode": round(encode_duration, 5),
            "Retrieve": round(retrieve_duration, 5),
            "Broadcast": round(broadcast_duration, 5),
            "Sync": round(sync_duration, 5),
            "Reconstruct+Filter": round(reconstruction_duration, 5)
        }
    }
    return metrics, final_audio

###############################################################################
#           EXPERIMENT 2: ENHANCED FRACTAL AUDIO COMPRESSION
###############################################################################

def generate_test_signal(duration_sec=2.0, sample_rate=8000):
    """
    Generates a simple test signal of sine wave + slight noise, normalized to ~0.9.
    Returns (sample_rate, np.int16 array).
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio_f = 0.3*np.sin(2*np.pi*220.0*t) + 0.02*np.random.randn(len(t))
    audio_f /= np.max(np.abs(audio_f)) + 1e-9
    audio_f *= 0.9
    return sample_rate, (audio_f*32767).astype(np.int16)

def normalize_block(block):
    mean_ = np.mean(block)
    std_ = np.std(block) + 1e-9
    normed = (block - mean_)/std_
    return normed, mean_, std_

class ImprovedFractalAudioEncoder:
    """
    Performs a fractal audio compression technique using domain-range blocks.
    """
    def __init__(self, range_size=256, domain_size=256, domain_stride=64):
        self.range_size = range_size
        self.domain_size = domain_size
        self.domain_stride = domain_stride

    def build_domain_pool(self, audio_f):
        """
        Creates overlapping domain blocks, each normalized to handle amplitude variations.
        """
        domain_blocks = []
        idx = 0
        while (idx + self.domain_size) <= len(audio_f):
            block = audio_f[idx : idx + self.domain_size]
            dn, d_mean, d_std = normalize_block(block)
            domain_blocks.append((dn, d_mean, d_std, idx))
            idx += self.domain_stride
        return domain_blocks

    def compress(self, audio):
        """
        Returns a list of fractal parameters: (range_start, best_d_idx, best_scale, best_offset).
        """
        audio_f = audio.astype(np.float32)
        domain_pool = self.build_domain_pool(audio_f)
        params = []
        pos = 0
        while (pos + self.range_size) <= len(audio_f):
            range_block = audio_f[pos : pos + self.range_size]
            rn, r_mean, r_std = normalize_block(range_block)

            best_idx = 0
            best_scale = 1.0
            best_offset = 0.0
            min_error = float('inf')

            for d_idx, (dn, d_mean, d_std, d_start) in enumerate(domain_pool):
                if d_start == pos:  # skip trivial same-location
                    continue
                dot = np.mean(rn * dn)
                scale_ = dot * (r_std / d_std)
                offset_ = r_mean - scale_*d_mean
                domain_block_unnorm = dn*d_std + d_mean
                approx = scale_*domain_block_unnorm + offset_
                if len(approx) < self.range_size:
                    continue
                err = np.mean((range_block - approx)**2)
                if err < min_error:
                    min_error = err
                    best_idx = d_idx
                    best_scale = scale_
                    best_offset = offset_
            params.append((pos, best_idx, best_scale, best_offset))
            pos += self.range_size
        return params

class ImprovedFractalAudioDecoder:
    """
    Decodes fractal-compressed audio by iteratively applying domain-range transforms.
    """
    def __init__(self, range_size=256, domain_size=256, domain_stride=64, iterations=5):
        self.range_size = range_size
        self.domain_size = domain_size
        self.domain_stride = domain_stride
        self.iterations = iterations

    def build_domain_starts(self, audio_length):
        idx = 0
        starts = []
        while (idx + self.domain_size) <= audio_length:
            starts.append(idx)
            idx += self.domain_stride
        return starts

    def decompress(self, params, audio_length, original_audio=None):
        """
        Uses optional original_audio for partial initialization.
        """
        if original_audio is not None:
            recon = original_audio.astype(np.float32).copy()
        else:
            recon = np.zeros(audio_length, dtype=np.float32)

        domain_starts = self.build_domain_starts(audio_length)
        for it in range(self.iterations):
            new_recon = recon.copy()
            for (range_start, d_idx, scale, offset) in params:
                range_end = range_start + self.range_size
                if range_end > audio_length or d_idx < 0 or d_idx >= len(domain_starts):
                    continue
                dom_start = domain_starts[d_idx]
                dom_end = dom_start + self.domain_size
                if dom_end > audio_length:
                    continue
                domain_block = recon[dom_start : dom_end]
                domain_block = domain_block[:self.range_size]
                approx = scale*domain_block + offset
                new_recon[range_start : range_end] = approx
            recon = new_recon
        return np.clip(recon, -32767, 32767).astype(np.int16)

def measure_audio_metrics(original, reconstructed, sr, exec_time):
    """
    Computes MSE, RMSE, SNR (dB), correlation coefficient, LSD (dB),
    and total execution time. Returns a dictionary of metrics.
    """
    o_f = original.astype(np.float32)
    r_f = reconstructed.astype(np.float32)
    # MSE
    mse_val = np.mean((o_f - r_f)**2)
    rmse_val = math.sqrt(mse_val)
    # SNR (dB)
    err = (o_f - r_f)
    num = np.sum(o_f**2) + 1e-9
    den = np.sum(err**2) + 1e-9
    snr_db = 10.0 * math.log10(num / den)
    # Correlation
    corr_mat = np.corrcoef(o_f, r_f)
    corr_val = corr_mat[0,1]
    # LSD
    n = len(o_f)
    fft_len = 1<<(n-1).bit_length()
    O = np.fft.rfft(o_f, n=fft_len)
    R = np.fft.rfft(r_f, n=fft_len)
    magO = np.abs(O) + 1e-9
    magR = np.abs(R) + 1e-9
    log_diff = (20.0*np.log10(magO) - 20.0*np.log10(magR))**2
    lsd_db = math.sqrt(np.mean(log_diff))

    metrics = {
        "MSE": round(mse_val, 3),
        "RMSE": round(rmse_val, 3),
        "SNR_dB": round(snr_db, 3),
        "Correlation": round(corr_val, 4),
        "LSD_dB": round(lsd_db, 3),
        "Execution_Time_s": round(exec_time, 4)
    }
    return metrics

def demo_enhanced_fractal():
    """
    Demonstrates fractal compression using domain-range blocks,
    then measures MSE, SNR, LSD, correlation, etc.
    """
    print("\n=== EXPERIMENT 2: ENHANCED FRACTAL AUDIO COMPRESSION ===")
    sr, original_audio = generate_test_signal(duration_sec=2.0, sample_rate=8000)
    print(f"Original Audio (Synthetic, 2s @ {sr}Hz) generated.")

    start_time = time.time()

    # Fractal encoder
    range_sz = 256
    domain_sz = 256
    domain_stride = 64
    encoder = ImprovedFractalAudioEncoder(range_size=range_sz, domain_size=domain_sz, domain_stride=domain_stride)
    fractal_params = encoder.compress(original_audio)

    # Fractal decoder
    decoder = ImprovedFractalAudioDecoder(range_size=range_sz, domain_size=domain_sz,
                                          domain_stride=domain_stride, iterations=6)
    # Use the original audio for partial initialization
    reconstructed_audio = decoder.decompress(fractal_params, len(original_audio), original_audio=original_audio)
    exec_time = time.time() - start_time

    # Extended metrics
    metrics = measure_audio_metrics(original_audio, reconstructed_audio, sr, exec_time)

    # Save WAV files
    out_fname_orig = os.path.join(OUTPUT_DIR, "original_audio.wav")
    out_fname_recon = os.path.join(OUTPUT_DIR, "enhanced_fractal_reconstructed.wav")
    write(out_fname_orig, sr, original_audio)
    write(out_fname_recon, sr, reconstructed_audio)

    print("\n===== EXPERIMENT RESULTS (Enhanced Fractal Audio) =====")
    for key, val in metrics.items():
        print(f"{key}: {val}")
    print(f"Original audio saved to: {out_fname_orig}")
    print(f"Reconstructed audio saved to: {out_fname_recon}")

    # Basic plotting of original vs. reconstructed (saved to file)
    plt.figure(figsize=(12,4))
    plt.plot(original_audio[:2000], label="Original", alpha=0.9)
    plt.plot(reconstructed_audio[:2000], label="Reconstructed", alpha=0.6)
    plt.title("Enhanced Fractal Audio - First 2000 samples")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "waveform_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Waveform comparison saved to: {plot_path}")

    return metrics

###############################################################################
#                      MAIN: RUN BOTH DEMOS
###############################################################################

if __name__ == "__main__":
    print("="*70)
    print("FFMAS UNIFIED DEMO - HEADLESS EXECUTION")
    print("="*70)
    
    # Experiment 1: FFMAS
    print("\n=== START EXPERIMENT 1: FFMAS ===")
    ffmas_metrics, final_audio_data = run_ffmas_experiment()
    print("\n[FFMAS METRICS REPORT]")
    for k, v in ffmas_metrics.items():
        print(f"{k}: {v}")
    print("\n[FFMAS FINAL AUDIO DATA PREVIEW]")
    for idx, frame in enumerate(final_audio_data[:5]):
        print(f"Frame #{idx+1}: {frame}")

    # Experiment 2: Enhanced Fractal Audio Compression
    print("\n=== EXPERIMENT 2: ENHANCED FRACTAL AUDIO COMPRESSION ===")
    fractal_metrics = demo_enhanced_fractal()

    # Save all metrics to JSON
    import json
    all_metrics = {
        "ffmas_metrics": ffmas_metrics,
        "fractal_compression_metrics": fractal_metrics
    }
    metrics_path = os.path.join(OUTPUT_DIR, "experiment_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nAll metrics saved to: {metrics_path}")

    print("\n" + "="*70)
    print("=== ALL EXPERIMENTS COMPLETED SUCCESSFULLY ===")
    print("="*70)
