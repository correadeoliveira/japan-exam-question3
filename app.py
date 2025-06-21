import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import torchvision.utils as vutils

# --- Definições do Modelo (DEVEM SER IDÊNTICAS às do script de treinamento) ---
# Parâmetros
latent_dim = 100
n_classes = 10
img_size = 28
channels = 1
img_shape = (channels, img_size, img_size)
device = torch.device("cpu") # O app pode rodar em CPU

# Arquitetura do Gerador
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

# --- Carregar o Modelo Treinado ---
@st.cache_resource
def load_model():
    model = Generator().to(device)
    # Carrega os pesos salvos
    model.load_state_dict(torch.load('generator.pth', map_location=device))
    model.eval() # Coloca o modelo em modo de avaliação
    return model

generator = load_model()

# --- Interface do Streamlit ---
st.title("Gerador de Dígitos Manuscritos")
st.write("Gere imagens sintéticas no estilo MNIST usando um modelo treinado do zero.") # 

# Seleção do dígito pelo usuário
digit_to_generate = st.selectbox(
    'Escolha um dígito para gerar (0-9)', # 
    options=list(range(10))
)

# Botão para gerar as imagens
if st.button('Gerar imagens'): # 
    st.subheader(f"Imagens geradas do dígito {digit_to_generate}") # 

    # Gerar 5 imagens
    num_images = 5

    # Prepara o ruído e os rótulos
    z = torch.randn(num_images, latent_dim).to(device)
    labels = torch.LongTensor([digit_to_generate] * num_images).to(device)

    with torch.no_grad():
        generated_imgs = generator(z, labels)

    # Re-normaliza as imagens de [-1, 1] para [0, 1] para exibição
    generated_imgs = (generated_imgs + 1) / 2.0

    # Exibe as imagens em 5 colunas
    cols = st.columns(num_images)
    for i, img_tensor in enumerate(generated_imgs):
        with cols[i]:
            # Converte o tensor para um formato que st.image entende
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            st.image(img_np, caption=f'Amostra {i+1}', width=100)