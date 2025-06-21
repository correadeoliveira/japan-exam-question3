import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import torchvision.utils as vutils

# --- Model Definitions (MUST BE IDENTICAL to the training script) ---
# Parameters
latent_dim = 100
n_classes = 10
img_size = 28
channels = 1
img_shape = (channels, img_size, img_size)
device = torch.device("cpu") # App can run on CPU

# Generator Architecture
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

# --- Load the Trained Model ---
@st.cache_resource
def load_model():
    model = Generator().to(device)
    # Load the saved weights
    model.load_state_dict(torch.load('generator.pth', map_location=device))
    model.eval() # Set the model to evaluation mode
    return model

generator = load_model()

# --- Streamlit Interface (in English) ---
st.title("Handwritten Digit Generator")
st.write("Generate synthetic MNIST-like images using a model trained from scratch.")

# User digit selection
digit_to_generate = st.selectbox(
    'Choose a digit to generate (0-9)',
    options=list(range(10))
)

# Button to generate images
if st.button('Generate images'):
    st.subheader(f"Generated images of digit {digit_to_generate}")
    
    # Generate 5 images
    num_images = 5
    
    # Prepare noise and labels
    z = torch.randn(num_images, latent_dim).to(device)
    labels = torch.LongTensor([digit_to_generate] * num_images).to(device)
    
    with torch.no_grad():
        generated_imgs = generator(z, labels)
    
    # Re-normalize images from [-1, 1] to [0, 1] for display
    generated_imgs = (generated_imgs + 1) / 2.0
    
    # Display images in 5 columns
    cols = st.columns(num_images)
    for i, img_tensor in enumerate(generated_imgs):
        with cols[i]:
            # Convert tensor to a displayable format for st.image
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            st.image(img_np, caption=f'Sample {i+1}', width=100)