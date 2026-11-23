import os
import torch
torch.set_grad_enabled(False)
torch.manual_seed(1234)
os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")
from .base import BaseModel
from .gemma3_replay import Gemma3Replay
from .gemma4_replay import Gemma4Replay
from .minicpm_v_4_5_replay import MiniCPM_V_4_5, MiniCPM_V_4_5_Replay
from .qwen3_vl import Qwen3VLChat
from .qwen35_vl_replay import Qwen35VLChatReplay
