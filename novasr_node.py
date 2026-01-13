"""
ComfyUI node wrapper for NovaSR (Audio Super Resolution)

This node wraps the NovaSR model to perform fast audio super-resolution
within the ComfyUI workflow. NovaSR is a tiny 50KB model that upscales
16kHz audio to 48kHz at 3600x realtime speed.
"""

import os
import sys
import random
import gc
from pathlib import Path
from urllib.request import urlretrieve
import urllib.error

import torch
import numpy as np
import soundfile as sf
import io

# Add parent directory to path so we can import NovaSR
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Try to import ComfyUI's folder_paths for model directory
try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False

# ComfyUI interrupt handling
try:
    from comfy import model_management
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

# NovaSR imports
from NovaSR.speechsr import SynthesizerTrn


def check_interrupted():
    """Check if ComfyUI processing was interrupted."""
    if HAS_COMFY:
        model_management.throw_exception_if_processing_interrupted()
    return False


def update_progress(current, total, node_prefix="NovaSR"):
    """Update ComfyUI progress bar if available."""
    if HAS_COMFY:
        try:
            state = model_management.get_progress_state()
            if state is not None:
                import comfy
                if hasattr(comfy, 'model_management'):
                    comfy.model_management.update_progress(
                        current / total if total > 0 else 0,
                        f"{node_prefix}: Processing {current}/{total}"
                    )
        except Exception:
            pass


# Global model cache to avoid reloading
_model_cache = None
_model_device = None
_model_path = None


def get_novasr_model_path():
    """Get the NovaSR models directory path."""
    if HAS_FOLDER_PATHS:
        try:
            models_dir = folder_paths.models_dir
            novasr_path = str(Path(models_dir) / "NovaSR")
            print(f"[NovaSR] Checking path: {novasr_path}")
            return novasr_path
        except (AttributeError, TypeError) as e:
            print(f"[NovaSR] folder_paths.models_dir failed: {e}")
            pass

    # Fallback: use local NovaSR directory
    fallback_path = str(Path(__file__).parent.parent / "models" / "NovaSR")
    print(f"[NovaSR] Using fallback path: {fallback_path}")
    return fallback_path


def download_novasr_model():
    """Download the NovaSR model if it doesn't exist."""
    model_dir = Path(get_novasr_model_path())
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "NovaSR.safetensors"
    
    if model_path.exists():
        print(f"[NovaSR] Model already exists at: {model_path}")
        return str(model_path)
    
    model_url = "https://huggingface.co/drbaph/NovaSR/resolve/main/NovaSR.safetensors?download=true"
    
    print(f"[NovaSR] Downloading model from {model_url}...")
    print(f"[NovaSR] Saving to: {model_path}")
    
    try:
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            print(f"[NovaSR] Downloading: {percent}% ({count * block_size / 1024 / 1024:.2f}MB / {total_size / 1024 / 1024:.2f}MB)")
        
        urlretrieve(model_url, model_path, reporthook=reporthook)
        print(f"[NovaSR] Model downloaded successfully!")
        return str(model_path)
    except urllib.error.URLError as e:
        print(f"[NovaSR] Error downloading model: {e}")
        return None
    except Exception as e:
        print(f"[NovaSR] Unexpected error downloading model: {e}")
        return None


class FastSRWrapper:
    """Wrapper for NovaSR with support for both .bin and .safetensors formats."""
    
    def __init__(self, ckpt_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hps = {
            "train": {
                "segment_size": 9600
            },
            "data": {
                "hop_length": 320,
                "n_mel_channels": 128
            },
            "model": {
                "resblock": "0",
                "resblock_kernel_sizes": [11],
                "resblock_dilation_sizes": [[1,3,5]],
                "upsample_initial_channel": 32,
            }
        }
        if ckpt_path is None:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id="YatharthS/NovaSR", filename="pytorch_model.bin", local_dir=".")

        self.model = self._load_model(ckpt_path).half().eval()

    def _load_model(self, ckpt_path):
        model = SynthesizerTrn(
            self.hps['data']['n_mel_channels'],
            self.hps['train']['segment_size'] // self.hps['data']['hop_length'],
            **self.hps['model']
        ).to(self.device)
        
        assert os.path.isfile(ckpt_path)
        
        # Load checkpoint based on file extension
        if ckpt_path.endswith(('.safetensors', '.sft')):
            # Load safetensors format
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path, device="cpu")
        else:
            # Load PyTorch format (.bin, .pth, .ckpt)
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            state_dict = checkpoint.get("state_dict", checkpoint)
        
        model.dec.remove_weight_norm()
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
    
    def infer(self, lowres_wav):
        with torch.no_grad():
            new_wav = self.model(lowres_wav)
        return new_wav.squeeze(0)


def load_novasr_model(ckpt_path, device="cuda"):
    """Load NovaSR model from checkpoint (supports both .bin and .safetensors)."""
    model = FastSRWrapper(ckpt_path=ckpt_path)
    model.model = model.model.to(device)
    return model


def generate_spectrogram_comparison(audio_before, sr_before, audio_after, sr_after=48000):
    """
    Generate a side-by-side spectrogram comparison image.

    Args:
        audio_before: Input audio (numpy array or tensor)
        sr_before: Input sample rate
        audio_after: Output audio (numpy array or tensor)
        sr_after: Output sample rate (always 48kHz for NovaSR)

    Returns:
        PIL Image: Side-by-side spectrogram comparison
    """
    print(f"[NovaSR] Spectrogram - Before: {sr_before/1000:.1f}kHz, After: {sr_after/1000:.1f}kHz")
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        from PIL import Image

        matplotlib.use('Agg')

        if isinstance(audio_before, torch.Tensor):
            audio_before = audio_before.cpu().numpy()
        if isinstance(audio_after, torch.Tensor):
            audio_after = audio_after.cpu().numpy()

        if audio_before.ndim > 1:
            audio_before = np.mean(audio_before, axis=0) if audio_before.shape[0] > 1 else audio_before[0]
        else:
            audio_before = audio_before.flatten()

        if audio_after.ndim > 1:
            audio_after = np.mean(audio_after, axis=0) if audio_after.shape[0] > 1 else audio_after[0]
        else:
            audio_after = audio_after.flatten()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a1a')
        ax1.set_facecolor('#000004')
        ax2.set_facecolor('#1a1a1a')
        fig.patch.set_alpha(1.0)
        ax1.set_alpha(1.0)
        ax2.set_alpha(1.0)

        import librosa
        import librosa.display

        D_before = librosa.amplitude_to_db(np.abs(librosa.stft(audio_before)), ref=np.max)

        img1 = librosa.display.specshow(
            D_before,
            sr=sr_before,
            hop_length=512,
            x_axis='time',
            y_axis='hz',
            cmap='magma',
            ax=ax1
        )
        ax1.set_title(f'Before ({sr_before/1000:.1f}kHz Input)', color='white', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Frequency (Hz)', color='white', fontsize=11)
        ax1.tick_params(axis='both', colors='white', labelsize=9)
        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.spines['right'].set_color('white')
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')
        ax1.set_ylim(0, 24000)

        D_after = librosa.amplitude_to_db(np.abs(librosa.stft(audio_after)), ref=np.max)
        img2 = librosa.display.specshow(
            D_after,
            sr=sr_after,
            hop_length=512,
            x_axis='time',
            y_axis='hz',
            cmap='magma',
            ax=ax2
        )
        ax2.set_title('After (NovaSR 48kHz)', color='white', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', color='white', fontsize=11)
        ax2.set_ylabel('Frequency (Hz)', color='white', fontsize=11)
        ax2.tick_params(axis='both', colors='white', labelsize=9)
        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.spines['right'].set_color('white')
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax2.set_ylim(0, 24000)

        cbar = fig.colorbar(img2, ax=[ax1, ax2], fraction=0.02, pad=0.04)
        cbar.set_label('dBFS', color='white', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=9)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        fig.suptitle('NovaSR Spectrogram Comparison',
                     color='white', fontsize=16, fontweight='bold')

        try:
            fig.set_layout_engine('constrained')
        except (AttributeError, TypeError):
            plt.tight_layout(rect=[0, 0, 1, 0.96])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#1a1a1a', dpi=200)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        return img

    except ImportError as e:
        print(f"[NovaSR] Warning: Could not import matplotlib for spectrogram: {e}")
        return None
    except Exception as e:
        print(f"[NovaSR] Warning: Could not generate spectrogram: {e}")
        return None


class NovaSRNode:
    """
    NovaSR Audio Super Resolution node for ComfyUI.

    Upscales audio to 48kHz using the ultra-fast NovaSR model.
    """

    DESCRIPTION = "Upscale audio to 48kHz using NovaSR (3600x realtime, 50KB model)"
    _logged_models = False

    @classmethod
    def INPUT_TYPES(cls):
        model_dir = get_novasr_model_path()
        model_files = []

        if not cls._logged_models:
            print(f"[NovaSR] Looking for models in: {model_dir}")

        if os.path.exists(model_dir):
            if not cls._logged_models:
                print(f"[NovaSR] Model directory exists!")
            for f in os.listdir(model_dir):
                if f.endswith(('.bin', '.safetensors')):
                    model_files.append(f)
                    if not cls._logged_models:
                        print(f"[NovaSR] Found model: {f}")
            if not cls._logged_models:
                print(f"[NovaSR] Total models found: {len(model_files)}")
        else:
            if not cls._logged_models:
                print(f"[NovaSR] Model directory does NOT exist!")

        if not model_files:
            if not cls._logged_models:
                print(f"[NovaSR] No models found, attempting auto-download...")
            downloaded_model = download_novasr_model()
            if downloaded_model and os.path.exists(downloaded_model):
                model_filename = os.path.basename(downloaded_model)
                model_files.append(model_filename)
                if not cls._logged_models:
                    print(f"[NovaSR] Successfully downloaded model: {model_filename}")

        if not model_files:
            model_files = ["pytorch_model.bin (download required)", "NovaSR.safetensors (download required)"]
            if not cls._logged_models:
                print(f"[NovaSR] Auto-download failed, using default options")

        cls._logged_models = True

        return {
            "required": {
                "audio": ("AUDIO", {}),
            },
            "optional": {
                "model": (model_files, {
                    "default": model_files[0] if model_files else "pytorch_model.bin (download required)",
                    "tooltip": "Model checkpoint file (place in ComfyUI/models/NovaSR/)"
                }),
                "output_stereo": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Convert output to stereo (mono input required for NovaSR, but some ComfyUI pipelines need stereo)"
                }),
                "unload_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload model from memory after generation (frees VRAM, but slower next run)"
                }),
                "show_spectrogram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate before/after spectrogram comparison image"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "IMAGE")
    RETURN_NAMES = ("audio", "spectrogram")
    OUTPUT_NODE = False
    FUNCTION = "upscale_audio"
    CATEGORY = "audio"

    def upscale_audio(
        self,
        audio: tuple,
        model: str = "pytorch_model.bin (download required)",
        output_stereo: bool = False,
        unload_model: bool = False,
        show_spectrogram: bool = True
    ):
        """
        Main processing function for audio super-resolution.

        Args:
            audio: ComfyUI audio tuple (audio_tensor, sample_rate) or dict
            model: Model checkpoint file name
            output_stereo: Convert output to stereo (mono input required for NovaSR)
            unload_model: Unload model from GPU memory after generation
            show_spectrogram: Generate before/after spectrogram comparison

        Returns:
            tuple: (audio, spectrogram) - ComfyUI audio format at 48kHz and optional spectrogram image
        """
        global _model_cache, _model_device, _model_path

        if isinstance(audio, dict):
            audio_waveform = audio['waveform']
            sr = audio['sample_rate']
        else:
            audio_waveform, sr = audio

        if isinstance(audio_waveform, str):
            raise ValueError(
                f"Audio input is a filename string ('{audio_waveform}'), not audio data. "
                f"Please connect the Load Audio node output to the NovaSR audio input."
            )

        original_sr = sr

        if isinstance(audio_waveform, torch.Tensor):
            audio_waveform = audio_waveform.cpu().numpy()
        elif not isinstance(audio_waveform, np.ndarray):
            raise TypeError(
                f"Audio waveform must be a torch.Tensor or numpy array, got {type(audio_waveform)}"
            )

        if audio_waveform.ndim == 1:
            audio_waveform = audio_waveform[np.newaxis, :]
        elif audio_waveform.ndim == 3:
            audio_waveform = audio_waveform.squeeze(0)

        original_audio_for_spec = audio_waveform.copy()
        original_sr_for_spec = sr
        print(f"[NovaSR] Captured original audio for spectrogram: sr={original_sr_for_spec}Hz, shape={original_audio_for_spec.shape}")

        is_stereo = audio_waveform.shape[0] > 1
        if is_stereo:
            print(f"[NovaSR] Stereo input detected, converting to mono (NovaSR is mono-only)")
            audio_waveform = np.mean(audio_waveform, axis=0)
            audio_waveform = audio_waveform[np.newaxis, :]

        if sr != 16000:
            import librosa
            print(f"[NovaSR] Resampling from {sr}Hz to 16000Hz (NovaSR requirement)")
            resampled_channels = []
            for ch in range(audio_waveform.shape[0]):
                channel = audio_waveform[ch]
                resampled = librosa.resample(channel, orig_sr=sr, target_sr=16000)
                resampled_channels.append(resampled)
            audio_waveform = np.stack(resampled_channels, axis=0)
            sr = 16000

        num_samples = audio_waveform.shape[1]
        duration_sec = num_samples / sr

        print(f"[NovaSR] Processing audio: {duration_sec:.2f}s at {sr}Hz")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if "download required" in model:
            raise ValueError(
                f"Model not found. Please download the NovaSR model and place it in:\n"
                f"{get_novasr_model_path()}\n\n"
                f"Download from: https://huggingface.co/YatharthS/NovaSR"
            )

        ckpt_path = os.path.join(get_novasr_model_path(), model)
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Model file not found: {ckpt_path}")

        if _model_cache is None or _model_device != device or _model_path != ckpt_path:
            print(f"[NovaSR] Loading model '{model}' on {device}...")
            _model_cache = load_novasr_model(ckpt_path, device)
            _model_device = device
            _model_path = ckpt_path
        else:
            print(f"[NovaSR] Using cached model on {device}")

        mm = model_management if HAS_COMFY else None

        try:
            with torch.no_grad():
                processed_channels = []

                for ch_idx in range(audio_waveform.shape[0]):
                    check_interrupted()

                    channel = audio_waveform[ch_idx]
                    lowres_wav = torch.from_numpy(channel).unsqueeze(0).half().unsqueeze(1).to(device)

                    print(f"[NovaSR] Upsampling channel {ch_idx + 1}/{audio_waveform.shape[0]}")

                    update_progress(ch_idx, audio_waveform.shape[0])

                    new_wav = _model_cache.infer(lowres_wav)
                    new_wav = new_wav.squeeze(0).cpu()

                    if isinstance(new_wav, torch.Tensor):
                        new_wav = new_wav.numpy()

                    if new_wav.ndim == 1:
                        new_wav = new_wav[np.newaxis, :]

                    processed_channels.append(new_wav)

                output_waveform = np.concatenate(processed_channels, axis=0)

                output_waveform = torch.from_numpy(output_waveform).float()

                if output_waveform.ndim == 1:
                    output_waveform = output_waveform.unsqueeze(0)

                assert output_waveform.ndim == 2, f"Output waveform must be 2D [channels, samples], got shape {output_waveform.shape}"

                if output_stereo and output_waveform.shape[0] == 1:
                    print(f"[NovaSR] Converting mono to stereo output...")
                    output_waveform = output_waveform.repeat(2, 1)

                output_waveform = output_waveform.unsqueeze(0)
                print(f"[NovaSR] Processing complete! Output: {output_waveform.shape[-1]/48000:.2f}s at 48kHz, shape: {output_waveform.shape}")

                if unload_model:
                    print("[NovaSR] Unloading model from VRAM...")
                    if _model_cache is not None:
                        del _model_cache
                        _model_cache = None
                        _model_device = None
                        _model_path = None
                    gc.collect()
                    if mm is not None:
                        mm.soft_empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("[NovaSR] Model unloaded, VRAM freed")

                spectrogram_image = None
                if show_spectrogram:
                    waveform_for_spec = output_waveform.squeeze(0)
                    spectrogram_image = generate_spectrogram_comparison(
                        original_audio_for_spec,
                        original_sr_for_spec,
                        waveform_for_spec,
                        48000
                    )

                if spectrogram_image is not None:
                    spectrogram_image = spectrogram_image.convert('RGB')
                    img_array = np.array(spectrogram_image).astype(np.float32) / 255.0
                    spectrogram_tensor = torch.from_numpy(img_array)
                    spectrogram_tensor = spectrogram_tensor.unsqueeze(0)
                    spectrogram_tensor = torch.clamp(spectrogram_tensor, 0.0, 1.0)
                else:
                    spectrogram_tensor = torch.zeros((1, 256, 256, 3))

                audio_output = {"waveform": output_waveform, "sample_rate": 48000}
                return (audio_output, spectrogram_tensor)

        except Exception as e:
            if HAS_COMFY:
                try:
                    from comfy.model_management import InterruptProcessingException
                    if isinstance(e, InterruptProcessingException):
                        if unload_model and _model_cache is not None:
                            del _model_cache
                            _model_cache = None
                            _model_device = None
                            _model_path = None
                        gc.collect()
                        if mm is not None:
                            mm.soft_empty_cache()
                        raise
                except ImportError:
                    pass
            raise

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        model = kwargs.get("model", "")
        output_stereo = kwargs.get("output_stereo", False)
        return float(f"{hash(model + str(output_stereo))}")


def register_folder_paths():
    """Register the NovaSR models folder with ComfyUI."""
    try:
        import folder_paths
        folder_paths.add_model_folder_path("novasr", "NovaSR")
    except Exception:
        pass


NODE_CLASS_MAPPINGS = {
    "NovaSR": NovaSRNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NovaSR": "NovaSR"
}

register_folder_paths()
