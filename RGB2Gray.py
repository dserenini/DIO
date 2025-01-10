from PIL import Image

def show_images(img_original, img_gray, img_binarized):
    img_original.show(title="Imagem Original")
    img_gray.show(title="Imagem em Escala de Cinza")
    img_binarized.show(title="Imagem Binarizada")


def img2grayscaleBW(input_file, gray_output_file, bin_output_file, threshold=128):
    # Abre a imagem original
    img_original = Image.open(input_file).convert("RGB")
    width, height = img_original.size
    pixels_original = img_original.load()  # Carrega os pixels

    # Criação da imagem em escala de cinza
    img_gray = img_original.copy()
    pixels_gray = img_gray.load()

    # Criação da imagem binarizada
    img_binarized = img_original.copy()
    pixels_binarized = img_binarized.load()

    for y in range(height):
        for x in range(width):
            r, g, b = pixels_original[x, y]  # Pega os valores RGB do pixel
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)  # Calcula o valor de escala de cinza
            pixels_gray[x, y] = (gray_value, gray_value, gray_value)  # Define pixel em escala de cinza

            # Binarização: 255 (branco) se acima do threshold, 0 (preto) caso contrário
            binary_value = 255 if gray_value > threshold else 0
            pixels_binarized[x, y] = (binary_value, binary_value, binary_value)  # Define pixel binarizado

    # Salvar as imagens em escala de cinza e binarizada
    img_gray.save(gray_output_file)
    img_binarized.save(bin_output_file)

    print(f"Imagem em escala de cinza salva como '{gray_output_file}'.")
    print(f"Imagem binarizada salva como '{bin_output_file}'.")

    # Exibe as imagens
    show_images(img_original, img_gray, img_binarized)


# Exemplo de uso:
input_path = "C:/Users/Danilo/Downloads/Lena.png"
gray_output_path = "C:/Users/Danilo/Downloads/Lena_grayscale.png"
bin_output_path = "C:/Users/Danilo/Downloads/Lena_binarized.png"

img2grayscaleBW(input_path, gray_output_path, bin_output_path)
