from PIL import Image
import os

# Caminho da pasta onde estão as imagens .webp
input_folder = './img'
# Caminho da pasta onde as imagens .png serão salvas
output_folder = './png'

# Verifica se a pasta de saída existe, se não, cria a pasta
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Função para converter WEBP para PNG
def convert_webp_to_png(input_path, output_path):
    for filename in os.listdir(input_path):
        if filename.endswith(".webp"):
            webp_image = Image.open(os.path.join(input_path, filename))
            # Remove a extensão .webp e substitui por .png
            output_file = os.path.splitext(filename)[0] + ".png"
            webp_image.save(os.path.join(output_path, output_file), "PNG")
            print(f"Imagem convertida: {filename} -> {output_file}")

# Converter todas as imagens .webp na pasta para .png
convert_webp_to_png(input_folder, output_folder)
