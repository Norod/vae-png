# vae-png
An experiment in encapsulating VAE encoded image data in a png file and decoding it back to the original image

```python
from vaepng import VaeImage
vaeimage = VaeImage()
vaeimage.load_vae("madebyollin/sdxl-vae-fp16-fix")

#VAE Encode Example00006 and encapsulate its latents in Example00006_VAE
vaeimage.img2vaepng(input_file = "test_images/Example00006.jpg", output_file = "Example00006_VAE.png")

#Restore VAE latents from Example00006_VAE, VAE decode them and save the result as Example00006_VAE_decoded
vaeimage.vaepng2img(input_file = "Example00006_VAE.png", output_file = "Example00006_VAE_decoded.jpg")

```
![vae-png-example](https://github.com/Norod/vae-png/assets/3617152/f63378fd-c13e-4053-87a9-2007ef2b53b2)
