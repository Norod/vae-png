# vae-png
An experiment in encapsulating VAE encoded image data in a png file and decoding it back to the original image

### Usage example

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

### Added exif data

Note that there are two fields added to the png's exif data.
In order to re-expand the 4 X 8 bit per channel PNG data back to 4 X 32 bit per channel latent data, the original Gobal Minimum and Maximum values from the 32bit latents are stored.

![MinMax_Exif](https://github.com/Norod/vae-png/assets/3617152/c654cc72-beeb-44ed-8168-6df47b2724bb)
