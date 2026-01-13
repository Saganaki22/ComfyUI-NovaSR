# ComfyUI-NovaSR

Ultra-fast audio super resolution custom node for ComfyUI, powered by the NovaSR model.

<p align="center">
  <a href="https://huggingface.co/drbaph/NovaSR/tree/main">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model">
  </a>
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  </a>
  <a href="https://github.com/comfyanonymous/ComfyUI">
    <img src="https://img.shields.io/badge/ComfyUI-Custom%20Node-orange.svg" alt="ComfyUI">
  </a>
</p>

<img width="1515" height="1012" alt="image" src="https://github.com/user-attachments/assets/a10cab1b-020c-4e4c-b326-b9cea0644dde" />




https://github.com/user-attachments/assets/f91bfa74-e390-4106-9f36-6df6e9a7b405



## Features

- **Incredibly Fast**: 3600x realtime speed on a single A100 GPU
- **Tiny Model**: Just 50KB in size
- **High Quality**: On par with models 5,000x larger
- **Simple to Use**: Just one click to upscale audio to 48kHz
- **Auto Stereo→Mono**: Automatically converts stereo audio to mono
- **Stereo Output Toggle**: Option to output stereo for ComfyUI compatibility (duplicates mono)
- **Any Audio Format**: Works with all formats ComfyUI supports (wav, mp3, flac, ogg, etc.)
- **Smart Resampling**: Automatically resamples to required 16kHz input

## Installation

### Via ComfyUI Manager (Recommended)

1. Open ComfyUI
2. Click the **Manager** button → **Install Custom Nodes**
3. Search for **"NovaSR"**
4. Click **Install** on "ComfyUI-NovaSR"
5. Restart ComfyUI

### Manual Installation

1. Clone or copy this repository into your ComfyUI `custom_nodes` directory:
   ```
   ComfyUI/custom_nodes/ComfyUI-NovaSR/
   ```

2. Install dependencies:
   ```bash
   pip install -r ComfyUI/custom_nodes/ComfyUI-NovaSR/requirements.txt
   ```

3. Download the NovaSR model and place it this location:
   - `ComfyUI/models/NovaSR/`

   Download from: https://huggingface.co/drbaph/NovaSR/tree/main
   
   Place one of these files:
   - `pytorch_model.bin`
   - `NovaSR.safetensors`

## Usage

1. Start ComfyUI
2. Right-click → Add Node → Audio → **NovaSR**
3. Load audio using **Load Audio** node (supports: wav, mp3, flac, ogg, aiff, and more)
4. Connect audio output to the NovaSR node input
5. The node will automatically:
    - Accept any sample rate audio input
    - Resample to 16kHz (NovaSR requirement)
    - Convert stereo to mono if needed (NovaSR is mono-only)
    - Upscale to 48kHz at 3600x realtime speed
    - Output a before/after spectrogram comparison image

**Note:** NovaSR outputs mono audio by default. Use the **Output Stereo** toggle to convert to stereo for ComfyUI compatibility.

## Node Parameters

- **Audio**: Input audio (required)
  - Works with any audio format supported by ComfyUI's Load Audio node
  - Automatically accepts any sample rate
  - Converts stereo to mono automatically
- **Model**: Model file to use (optional, defaults to available model)
  - Supports `.bin` and `.safetensors` formats
- **Output Stereo**: Convert mono output to stereo (optional, default: False)
  - Duplicates mono channel for ComfyUI pipelines that require stereo input
- **Unload Model**: Remove model from VRAM after processing (optional, default: False)
  - Frees GPU memory but slower on next run
- **Show Spectrogram**: Generate before/after spectrogram comparison (optional, default: True)
  - Visualizes frequency improvements

## Audio Format Support

NovaSR works with any audio format that ComfyUI supports:

**Supported Formats:**
- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- OGG Vorbis (.ogg)
- AIFF (.aiff)
- And more via ComfyUI's Load Audio node

**Audio Processing:**
- **Input Sample Rate**: Any (automatically resampled to 16kHz)
- **Input Channels**: Mono or Stereo (stereo automatically mixed to mono)
- **Output Sample Rate**: 48kHz (fixed)
- **Output Channels**: Mono (default) or Stereo (when toggle enabled)

**Tip:** Enable **Output Stereo** toggle if your ComfyUI pipeline requires stereo audio. This duplicates the mono channel to both left and right channels.

## How It Works

NovaSR is a revolutionary tiny model (only 50KB!) that can upscale muffled 16kHz audio into clear and crisp 48kHz audio. It uses:
- Less than 10 tiny Conv1D layers
- Snake activations based on BigVGAN
- Extreme efficiency optimizations

## Comparison with Other Models

| Model | Speed (Real-Time) | Model Size |
|-------|------------------|------------|
| **NovaSR** | **3600x** | **48-53 KB** |
| FlowHigh | 20x | ~450 MB |
| FlashSR | 14x | ~1000 MB |
| AudioSR | 0.6x | ~2000 MB |

## Use Cases

- Enhancing TTS model quality with nearly 0 computational cost
- Real-time enhancement of low-quality calls and audio
- Restoring audio datasets
- Improving voice quality in recordings

## Credits

- ComfyUI-NovaSR Plugin: https://github.com/Saganaki22/ComfyUI-NovaSR
- Model on Hugging Face: https://huggingface.co/drbaph/NovaSR
- Original NovaSR by Yatharth Sharma: https://github.com/ysharma3501/NovaSR

## License

This project follows the license of the original NovaSR repository.
