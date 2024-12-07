import importlib
import torch
import json

# Hardcoded values
evaluation_name = "visualization"
output_dir = "your_output_directory"
config_path = r"D:\Python\Progetti\hackaton_progan\v1\pytorch_GAN_zoo\output_networks\default\default_train_config.json"
checkpoint_path = r"D:\Python\Progetti\hackaton_progan\v1\pytorch_GAN_zoo\output_networks\default\default_s6_i400000.pt"

# Load the PGAN model
model_module = importlib.import_module("models.progressive_gan_custom")
ProgressiveGAN = model_module.ProgressiveGAN

# Load the configuration from JSON
with open(config_path, 'r') as f:
    config = json.load(f)

pgan = ProgressiveGAN(useGPU=True,                 
                dimLatentVector=128,
                 depthScale0=256,
                 initBiasToZero=True,
                 leakyness=0.2,
                 perChannelNormalization=True,
                 miniBatchStdDev=False,
                 equalizedlR=True,
                 )
    

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Prepare the state dictionary for loading
state_dict = {
    'netG': checkpoint['netG'],
    'config': config  # Add the config to match the expected structure
}

# Load the state dictionary
pgan.load_state_dict(state_dict)

# Generate an image from a latent vector of size 128
latent_vector = torch.randn(1, 128).to(pgan.device)
with torch.no_grad():
    generated_image = pgan.test(input=latent_vector, getAvG=True)
# pgan(latent_vector)
from PIL import Image
import numpy as np

# Convert the generated image tensor to a numpy array
generated_image_np = generated_image.squeeze().cpu().numpy()

# Normalize the image to the range [0, 255]
generated_image_np = (generated_image_np - generated_image_np.min()) / (generated_image_np.max() - generated_image_np.min()) * 255
generated_image_np = generated_image_np.astype(np.uint8)

# Convert the numpy array to a PIL Image
image = Image.fromarray(generated_image_np.transpose(1, 2, 0))

# Display the image
image.show()



# Save or visualize the generated image
print("Generated image shape:", generated_image.shape)
