"""Stage 2: SAM 3 text-prompted verification on a person crop.

SAM 3 is called in-process on the crop produced by Stage 1. The processor's
internal confidence filter (default 0.5) is set to 0 so we always get the raw
best score; the decision threshold is chosen on the dev split and then frozen
(see analyze.py), instead of relying on the library default.

Inference runs under CUDA autocast (without it SAM 3 raises "mat1 and mat2
must have the same dtype"). Meta's examples use bfloat16, but the Tesla T4 has
no native bfloat16 support: on T4 we measured ~3.2 s per crop with bfloat16 and
~1.25 s with float16, with the same score (0.94 vs 0.94) on a test image.
float16 is therefore the default; pass amp_dtype=torch.bfloat16 on Ampere+ GPUs.
"""
import numpy as np
import torch
from PIL import Image


class Sam3Verifier:
    def __init__(self, checkpoint, prompt="cigarette", device=None, amp_dtype=torch.float16):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt = prompt
        self.amp_dtype = amp_dtype
        model = build_sam3_image_model(checkpoint_path=checkpoint,
                                       device=self.device, load_from_HF=False)
        self.processor = Sam3Processor(model, device=self.device,
                                       confidence_threshold=0.0)

    @torch.inference_mode()
    def score(self, crop_bgr):
        """Return (best_score, best_box_xyxy or None) for the text prompt."""
        pil = Image.fromarray(np.ascontiguousarray(crop_bgr[:, :, ::-1]))
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype,
                            enabled=self.device.startswith("cuda")):
            state = self.processor.set_image(pil)
            out = self.processor.set_text_prompt(prompt=self.prompt, state=state)
        scores, boxes = out.get("scores"), out.get("boxes")
        if isinstance(scores, torch.Tensor):
            scores = scores.float().cpu().numpy()
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.float().cpu().numpy()
        if scores is None or len(scores) == 0:
            return 0.0, None
        i = int(np.argmax(scores))
        return float(scores[i]), boxes[i]
