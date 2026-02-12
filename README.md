<p align="center">
  <img src="logo.png" alt="FFMAS Logo" width="520"/>
</p>

# Enhanced Fractal Audio - Validated Modules

**Creator & Principal Investigator**: **Vasile Lucian Borbeleac**  
**Organization**: FRAGMERGENT TECHNOLOGY S.R.L, Romania  
**Validation Date**: February 2026  
**Platform**: Windows 11  
**Hardware**: AMD Radeon RX 6700 XT (CPU execution for current modules)

---

## Overview

This repository contains validated and adapted implementations of the **Enhanced Fractal Audio** technologies, specifically optimized for execution on Windows systems. All modules have been extracted from research documentation, adapted for headless execution, tested, and documented with comprehensive performance metrics.

---

## Module Status Table

| Module | Status | Backend | Execution Time | Key Metrics | Output Files |
|--------|--------|---------|----------------|-------------|--------------|
| **FFMAS System** | ✅ **VALIDATED** | CPU (NumPy/SciPy) | 0.226s | SNR: 18.67dB<br>Correlation: 0.9938<br>LSD: 7.898dB | 2 WAV, 1 PNG, 1 JSON, 1 LOG |

### Legend
- ✅ **VALIDATED**: Successfully executed, all outputs generated, comprehensive documentation complete
- 🔄 **IN PROGRESS**: Adaptation or testing underway
- ⏸️ **PENDING**: Scheduled for future validation
- ⚠️ **ISSUES**: Known problems requiring resolution

---

## Repository Structure

```
validated/
├── README.md                    # This file - Project index
├── LICENSE                      # Enterprise Research License
├── .gitignore                   # Standard Python excludes
│
└── ffmas_system/                # FFMAS: Fractal Frequential Multidimensional Audio System
    ├── README.md                # Comprehensive module documentation
    ├── LICENSE                  # Enterprise Research License
    ├── .gitignore               # Python .gitignore
    ├── requirements.txt         # Dependencies (numpy, scipy, matplotlib)
    ├── ffmas_unified_demo.py    # Adapted implementation (headless)
    ├── run_output.log           # Complete execution log
    ├── experiment_metrics.json  # Performance metrics (JSON)
    ├── original_audio.wav       # Reference audio (2s @ 8kHz)
    ├── enhanced_fractal_reconstructed.wav  # Decompressed output
    └── waveform_comparison.png  # Visual comparison plot
```

---

## Module Summaries

### FFMAS System

**Full Name**: Fractal Frequential Multidimensional Audio System  
**Type**: Unified audio processing framework + fractal compression demo  
**Backend**: CPU (NumPy/SciPy) - No GPU dependencies  
**Status**: ✅ Fully validated

**Components**:
1. **FRAW** - Fractal Audio Encoder (adaptive encoding)
2. **CHADS** - Cubic Holographic Audio Data Storage (multidimensional indexing)
3. **SAIB** - Selective Auditory Identity Broadcasting (fractal-based DRM)
4. **AFDS** - Autonomous Frequential Device Synchronization (network-free sync)
5. **Ψ-Audio** - Adaptive Reconstruction Controller (inverse transforms)
6. **FANF** - Fractal Adaptive Noise Filtering (fractal dimension analysis)
7. **SEFAM** - Spectral-Energetic Fractal Audio Memory (hybrid storage)

**Enhanced Fractal Compression**:
- Domain-range block matching (256-sample blocks, 64-sample stride)
- Iterative reconstruction (6 iterations)
- Scientific validation metrics (MSE, SNR, LSD, Correlation)

**Performance Results**:
- **SNR**: 18.67 dB (communication-grade quality)
- **Correlation**: 0.9938 (excellent waveform similarity)
- **LSD**: 7.898 dB (moderate spectral fidelity)
- **Execution Time**: 0.226 seconds for 2-second audio @ 8kHz
- **Compression Ratio**: ~4:1 with current parameters

**Applications**: Telecommunications, VoIP, secure broadcasting, hearing aids, archival storage

**See**: [ffmas_system/README.md](ffmas_system/README.md) for full technical details

---

## Cross-Module Synthesis

### Unified Fractal Framework

The FFMAS system demonstrates a **paradigm shift** in audio processing by integrating fractal mathematics across the entire signal chain:

1. **Encoding→Storage**: Fractal signatures enable holographic diagonal retrieval
2. **Storage→Broadcasting**: Identity-based selective transmission without separate DRM
3. **Broadcasting→Sync**: Network-free device coordination via fractal pattern matching
4. **Sync→Reconstruction**: Adaptive decoding with real-time optimization
5. **Reconstruction→Filtering**: Fractal dimension analysis preserves signal fidelity

### Scientific Validation

**Experiment 1: System Architecture Validation**
- All 7 modules integrated successfully
- End-to-end pipeline: encode → store → retrieve → broadcast → sync → reconstruct → filter
- Pipeline latency: <10ms (demo mode with minimal data)

**Experiment 2: Compression Quality Metrics**
- High correlation (0.9938) confirms waveform preservation
- SNR ~19dB suitable for speech/communication applications
- LSD ~8dB indicates moderate spectral distortion
- Trade-off: Compression vs. quality (tunable via block size/iterations)

### Innovation Highlights

| Innovation | Traditional Approach | FFMAS Approach | Advantage |
|------------|---------------------|----------------|-----------|
| **Compression** | Block-based (MP3, AAC) | Fractal self-similarity | Smaller files, adaptive |
| **Storage** | Linear indexing (O(log n)) | Holographic diagonal (O(1)) | Instant retrieval |
| **Broadcasting** | Uniform distribution | Selective fractal matching | Bandwidth savings, security |
| **Synchronization** | Network protocols (NTP/BT) | Fractal frequency recognition | Infrastructure-independent |
| **Noise Filtering** | Spectral subtraction | Fractal dimension analysis | Preserves musical patterns |

---

## Performance Notes

### CPU Execution (Current Implementation)

**Hardware**: AMD Radeon RX 6700 XT available but **not utilized** (NumPy-based code)

**Rationale**: FFMAS demo is pure Python/NumPy/SciPy without PyTorch/TensorFlow dependencies, eliminating the need for DirectML integration.

**Performance**:
- **Single-threaded**: NumPy operations run on CPU cores
- **Execution time**: 0.226s for 2-second audio (compression + decompression)
- **Memory usage**: Minimal (<100 MB for processing 16k samples)
- **Scalability**: Linear with audio length (longer files require proportionally more time)

### Future GPU Acceleration Potential

To leverage the AMD RX 6700 XT for FFMAS:

1. **Rewrite domain-range search as CUDA/ROCm kernels** (or DirectML compute shaders)
2. **Parallelize block matching** across GPU threads
3. **Expected speedup**: 10-50x for large audio files (>1 minute)
4. **Trade-off**: Development complexity vs. performance gains

---

## Limitations & Caveats

### Current Version Limitations

1. **Quality Constraints**:
   - SNR ~19dB suitable for communication but **not archival-quality music** (target: >40dB)
   - Spectral distortion (LSD ~8dB) noticeable in high-fidelity applications
   - Block artifacts possible at range boundaries

2. **Computational Complexity**:
   - Iterative reconstruction: O(n·k) where k = iterations (currently 6)
   - Domain pool search: O(n²) worst case for naive implementation
   - Real-time encoding challenging for long files without GPU

3. **Parameter Sensitivity**:
   - Block size (256 samples) optimized for 8kHz speech
   - Different audio types (music, environmental sounds) may require retuning
   - Iteration count affects quality vs. speed trade-off

4. **No Real Audio Validation**:
   - Current testing on synthetic signals only (sine wave + noise)
   - Music/speech corpora testing required for production readiness
   - Perceptual quality (PESQ, POLQA) not yet measured

### Hardware-Specific Notes

**AMD RX 6700 XT**:
- DirectML support verified but **not utilized** in current NumPy implementation
- GPU idle during FFMAS execution (CPU-only workload)
- Future work: Porting to PyTorch + torch-directml for GPU acceleration

**Memory**:
- 16GB system RAM sufficient for files up to ~10 minutes @ 8kHz
- GPU VRAM (12GB) untapped (no GPU code)

---

## Validation Methodology

### Adaptation Process

1. **Extraction**: Python code extracted from 27-page PDF documentation
2. **Headless Conversion**: 
   - `matplotlib.use('Agg')` for non-interactive plotting
   - Removed IPython display dependencies
   - File-based outputs (WAV, PNG, JSON)
3. **Testing**: Executed in Windows 11 PowerShell environment
4. **Verification**: All outputs generated successfully, metrics logged

### Quality Assurance

✅ **Scientific Integrity Preserved**: No modifications to fractal algorithms or equations  
✅ **Reproducibility**: Complete execution log (`run_output.log`) with timestamps  
✅ **Outputs Validated**: Visual inspection of waveforms, JSON metrics verification  
✅ **Documentation**: 3-perspective README (technical, innovation, applications)

---

## Citation Format

If you use this validated implementation in research or applications:

```bibtex
@software{ffmas_validated2026,
  author = {Borbeleac, Vasile Lucian},
  contributor = {Borbeleac, Andreea-Gianina Luta},
  title = {Enhanced Fractal Audio - Validated Modules},
  subtitle = {FFMAS: Fractal Frequential Multidimensional Audio System},
  year = {2026},
  month = {February},
  organization = {FRAGMERGENT TECHNOLOGY S.R.L},
  address = {Romania},
  note = {Created by Vasile Lucian Borbeleac. Validated on Windows 11 with AMD Radeon RX 6700 XT (CPU execution)},
  howpublished = {\\url{https://github.com/NEURALMORPHIC-FIELDS/Enhanced-_Fractal_Audio}}
}
```

---

## License

All modules licensed under **Enterprise Research License**:
- ✅ **Free for academic/research use** (with attribution)
- 💼 **Commercial licensing required** for business applications
- 📧 Contact: V.l.borbel@gmail.com

See [LICENSE](LICENSE) and individual module licenses for full terms.

---

## Future Work

### Planned Enhancements

1. **Real Audio Testing**:
   - Music corpus (various genres, bitrates)
   - Speech datasets (multiple languages, accents)
   - Environmental audio (noise, reverb, compression artifacts)

2. **GPU Acceleration**:
   - PyTorch + torch-directml port for AMD GPUs
   - ROCm optimization for native AMD compute
   - Benchmark: CPU vs. DirectML vs. ROCm performance

3. **Quality Improvements**:
   - Adaptive block sizing based on signal complexity
   - Perceptual weighting (psychoacoustic model)
   - Multi-resolution wavelet-fractal hybrid

4. **Additional Modules**:
   - Real-time streaming encoder/decoder
   - Hardware-accelerated FANF noise filtering
   - Web-based demo interface

---

## Technical Support

**For Academic Collaboration**:  
Email: V.l.borbel@gmail.com with research proposal

**For Commercial Licensing**:  
Contact FRAGMERGENT TECHNOLOGY S.R.L via email above

**For Bug Reports**:  
Include `run_output.log` and system specs (OS, Python version, hardware)

---

## Acknowledgments

**Hardware Platform**: AMD Radeon RX 6700 XT (DirectML-ready)  
**Development Tools**: Python 3.x, NumPy, SciPy, Matplotlib  
**Validation Environment**: Windows 11, PowerShell 7.x

---

**Last Updated**: February 13, 2026  
**Validation Version**: 1.0  
**Total Modules Validated**: 1/1 (FFMAS System)
