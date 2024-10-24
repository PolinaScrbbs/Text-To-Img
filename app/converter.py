import os
import random
import string
import matplotlib.pyplot as plt
import keras_cv

MEDIA_FOLDER = "media"

async def converter(text: str):
    # Загружаем модель Stable Diffusion
    model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

    # Генерируем изображение на основе текста
    images = model.text_to_image(text, batch_size=1)

    # Создаём уникальное имя файла для каждого изображения
    unique_suffix = "".join(random.choices(string.digits, k=4))
    filename = os.path.join(MEDIA_FOLDER, f"image_{unique_suffix}.png")

    # Функция для отображения и сохранения изображений
    def plot_and_save_images(images, filename):
        plt.figure(figsize=(20, 20))
        for i in range(len(images)):
            ax = plt.subplot(1, len(images), i + 1)
            plt.imshow(images[i])
            plt.axis("off")

        # Сохранение изображения в файл
        plt.savefig(filename)

    # Вызываем функцию для отображения и сохранения изображений
    plot_and_save_images(images, filename)

    print(f"Изображение сохранено как: {filename}")
