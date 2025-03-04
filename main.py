import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model yang telah dilatih
model = load_model("intelligent-control-week3/cnn_model.h5")

# Direktori dataset
train_dir = "intelligent-control-week3/dataset/seg_train/seg_train"
val_dir = "intelligent-control-week3/dataset/seg_test/seg_test"

# Augmentasi data
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=32, class_mode='categorical')

# Load label kelas
class_labels = list(train_generator.class_indices.keys())

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Inisialisasi grafik
plt.ion()  # Aktifkan mode interaktif untuk update real-time
fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot(x_data, y_data, 'r-')  # Grafik berwarna merah
ax.set_ylim(0, 255)
ax.set_xlim(0, 50)  # Hanya menampilkan 50 frame terakhir
ax.set_xlabel("Frame")
ax.set_ylabel("Rata-rata Intensitas Night Vision")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mode Night Vision dengan konversi ke skala abu-abu
    night_vision = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night_vision = cv2.applyColorMap(night_vision, cv2.COLORMAP_JET)

    # Hitung rata-rata intensitas pencahayaan
    avg_intensity = np.mean(night_vision)

    # Preprocessing gambar untuk prediksi
    img = cv2.resize(frame, (150, 150))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediksi kelas
    pred = model.predict(img)
    label = class_labels[np.argmax(pred)]

    # Tambahkan nilai intensitas ke grafik
    frame_count += 1
    x_data.append(frame_count)
    y_data.append(avg_intensity)

    # Batasi jumlah data yang ditampilkan
    if len(x_data) > 50:
        x_data.pop(0)
        y_data.pop(0)

    line.set_xdata(x_data)
    line.set_ydata(y_data)
    ax.set_xlim(max(0, frame_count - 50), frame_count)  # Update batas sumbu x

    plt.draw()
    plt.pause(0.01)  # Update grafik setiap iterasi

    # Tampilkan hasil
    cv2.putText(frame, f'Class: {label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    cv2.imshow('Night Vision', night_vision)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Matikan mode interaktif setelah loop selesai
plt.show()  # Tampilkan grafik akhir
