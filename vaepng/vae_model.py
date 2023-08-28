import torch
from diffusers import AutoencoderKL

class VaeModel:
    def __init__(self):
        self.model_id = None
        self.vae = None        
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self, 
             model_id = "madebyollin/sdxl-vae-fp16-fix"):
        self.model_id = model_id
        if self.torch_device == "cuda":            
            self.vae = AutoencoderKL.from_pretrained(self.model_id, torch_dtype=torch.float16)
        else:
            self.vae = AutoencoderKL.from_pretrained(self.model_id)
            
        self.vae = self.vae.to(self.torch_device)
        