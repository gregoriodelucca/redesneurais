import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Verificar versão do TensorFlow
print(f"TensorFlow Version: {tf.__version__}")

# Caminhos para os dados
train_dir = "dataset/train"
val_dir = "dataset/validation"

# 1. Carregar o modelo pré-treinado MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Congelar os pesos do modelo base (não treinar essas camadas)
base_model.trainable = False

# 2. Adicionar camadas personalizadas no topo
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Camada de Pooling Global
x = Dense(128, activation="relu")(x)  # Camada totalmente conectada
predictions = Dense(2, activation="softmax")(x)  # Saída para 2 classes

# Modelo completo
model = Model(inputs=base_model.input, outputs=predictions)

# 3. Compilar o modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 4. Preparar os dados com aumento (augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

# Carregar os dados
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# 5. Treinar o modelo
history = model.fit(
    train_generator,
    epochs=10,  # Número de épocas (ajuste conforme necessário)
    validation_data=validation_generator
)

# 6. Plotar resultados
plt.figure(figsize=(12, 4))

# Gráfico de Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Gráfico de Perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.show()
