"""Release registry: reuse the native replay-only model entries."""
supported_VLM = {}
from .config_qwen_minimal import supported_VLM as _entries
supported_VLM.update(_entries)
from .config_minicpm45_minimal import supported_VLM as _entries
supported_VLM.update(_entries)
from .config_gemma3_minimal import supported_VLM as _entries
supported_VLM.update(_entries)
from .config_gemma4_minimal import supported_VLM as _entries
supported_VLM.update(_entries)
from .config_text_filler_minimal import supported_VLM as _entries
supported_VLM.update(_entries)
