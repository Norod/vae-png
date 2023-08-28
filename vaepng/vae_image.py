from .vae_model import VaeModel

import torch
from diffusers import AutoencoderKL
from torchvision import transforms as tfms

from PIL import Image
import numpy as np
from PIL.PngImagePlugin import PngInfo

class VaeImage:
    def __init__(self, 
                 vaemodel = VaeModel()):
        self.vaemodel = vaemodel
        self.torch_device = vaemodel.torch_device
        self.vae = vaemodel.vae
        self.to_tensor_tfm = tfms.ToTensor()

    def load_vae(self, model_id_or_path):
        self.vaemodel.load(model_id_or_path)
        self.vae = self.vaemodel.vae

    def encode_pil_to_latent(self, input_im):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        with torch.no_grad():
            latent = self.vae.encode(self.to_tensor_tfm(input_im).unsqueeze(0).to(self.torch_device)*2-1) # Note scaling
        return 0.18215 * latent.to_tuple()[0].mean

    def decode_latents_to_pil(self, latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        latents = latents.to(self.torch_device)
        with torch.no_grad():
            image = self.vae.decode(latents)
        image = (image.sample.detach().cpu() / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images


    def encode(self, input_image_file):
        # Load the image with PIL
        input_image = Image.open(input_image_file)
        encoded = self.encode_pil_to_latent(input_image)    
        return encoded

    def decode(self, encoded_latents):
        decoded = self.decode_latents_to_pil(encoded_latents)[0]    
        return decoded

    def vaepng2img(self, input_file, output_file):
        print("Load reduced latents")
        reduced_latents = VaeImage.reduced_latents_from_png(png_image_name=input_file)  
        print("Decode reduced latents")
        image = self.decode(reduced_latents)
        image.save(output_file)


    def img2vaepng(self, input_file, output_file):
        encoded_latents = self.encode(input_file)
        encoded_latents_as_image, minValue, maxValue = self.latents_as_images(encoded_latents)
        metadata = PngInfo()  #Write MinVale, MaxValue as part of the PNG metadata
        metadata.add_itxt("minValue", minValue)
        metadata.add_itxt("maxValue", maxValue)
        encoded_latents_as_image[0].save(output_file, pnginfo=metadata)  #Note: The alpha channel also contains information

    # Static utility methods

    @staticmethod
    def latents_as_images(latents):
        minValue = latents.min()
        maxValue = latents.max()
        latents = (latents-minValue)/(maxValue-minValue)  
        image = latents.detach().cpu().permute(0, 2, 3, 1).numpy()        
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        minValue = minValue.detach().cpu().numpy().astype("float32")
        maxValue = maxValue.detach().cpu().numpy().astype("float32")
        return pil_images, str(minValue), str(maxValue)

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except:
            return False

    @staticmethod
    def reduced_latents_from_png(png_image_name):
        image_in = Image.open(png_image_name)
        image_text_data = image_in.text         #Check the PNG metadata for minValue, maxValue keys and load their value as float32 if possible
        minValue = image_text_data["minValue"]
        if VaeImage.is_float(minValue):
            minValue = float(minValue)
        else:
            minValue = float(-3.5)
        maxValue = image_text_data["maxValue"]
        if VaeImage.is_float(maxValue):
            maxValue = float(maxValue)
        else:
            maxValue = float(4.0)
        image_in = np.array(image_in, np.float32)  
        print(image_in.shape)
        image_in = image_in/255.0
        image_in = image_in.transpose((2, 0, 1))
        image_out = np.expand_dims(image_in, 0)  
        reduced_latents = torch.tensor(image_out)
        reduced_latents = (reduced_latents*(maxValue-minValue))+minValue
        print(reduced_latents.shape)
        return reduced_latents
