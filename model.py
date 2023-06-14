import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv('train.txt', delimiter=';', header=None, names=['sentence','emotion'])


encoder = LabelEncoder()
df['emotion'] = encoder.fit_transform(df['emotion'])


sentences = df['sentence'].values
labels = df['emotion'].values

x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)


tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_train_padded = pad_sequences(x_train_seq, maxlen=20, truncating='post')


x_test_seq = tokenizer.texts_to_sequences(x_test)
x_test_padded = pad_sequences(x_test_seq, maxlen=20, truncating='post')


# set vocabulary size to the number of unique words in your data + 1 for <OOV>
vocab_size = len(tokenizer.word_index) + 1

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=20),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

history = model.fit(x_train_padded, y_train, epochs=5, 
                    validation_data=(x_test_padded, y_test), verbose=2)

# Menyimpan model
model.save('my_model.h5')

# Membuat grafik untuk akurasi
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Membuat grafik untuk loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

import json

# Menyimpan word index (vocab) dalam format json
with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=4)

# Menyimpan konfigurasi tokenizer dalam format json
tokenizer_json = tokenizer.to_json()
with open('tokenizer_config.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(json.loads(tokenizer_json), ensure_ascii=False))

# Menyimpan tokenizer dalam format json
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer.to_json())


# Menyimpan tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Menyimpan label encoder
with open('label_encoder.pickle', 'wb') as le:
    pickle.dump(encoder, le, protocol=pickle.HIGHEST_PROTOCOL)


# Memuat model dan tokenizer
model = load_model('my_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Memuat Label Encoder
with open('label_encoder.pickle', 'rb') as le:
    encoder = pickle.load(le)

def predict_emotion(text):
    # Mengubah text menjadi sequence
    sequences = tokenizer.texts_to_sequences([text])
    # Padding sequence
    padded_sequences = pad_sequences(sequences, maxlen=20, truncating='post')
    # Membuat prediksi
    prediction = model.predict(padded_sequences)
    # Mengambil label dengan probabilitas tertinggi
    predicted_label = encoder.classes_[np.argmax(prediction)]
    return predicted_label

# Simulasi chat bot
while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        break
    response = predict_emotion(user_input)
    print("ChatBot: ", response)
