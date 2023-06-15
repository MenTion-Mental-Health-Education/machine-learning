import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import random

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

vocab_size = len(tokenizer.word_index) + 1

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=20),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

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

def generate_response(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=20, truncating='post')
    prediction = model.predict(padded_sequences)
    predicted_label = encoder.classes_[np.argmax(prediction)]
    return predicted_label


def generate_alternative_response(emotion):
    if emotion == "fear":
        alternative_responses = [
            "Perhaps the feeling of fear arises due to uncertainty in the situation. You can try to find ways to address it by seeking more information.",
            "Fear is a natural response to challenging situations. Try to identify the root of the problem and seek support from your loved ones.",
            "Don't let fear hold you back. Face your fears bravely and find ways to overcome them gradually."
        ]
    elif emotion == "joy":
        alternative_responses = [
            "Joy is an incredible gift. Fully embrace and enjoy the moment, and share your happiness with your loved ones.",
            "Joy brings positive energy into our lives. Keep seeking activities and experiences that bring you happiness and gratitude.",
            "When you feel joyful, don't hesitate to spread happiness to others. Kindness and joy can inspire those around you."
        ]
    elif emotion == "anger":
        alternative_responses = [
            "When experiencing anger, try to pause and slowly manage it. Communicate calmly and search for better solutions.",
            "Anger is a natural emotion, but it's important to manage it effectively. Find ways to express your feelings calmly and constructively.",
            "Managing anger is a challenge, but you can do it. Take time to reflect, practice relaxation techniques, or talk to someone who can provide support."
        ]
    elif emotion == "sadness":
        alternative_responses = [
            "Sadness is a natural part of life. Allow yourself to feel it, but remember to seek support and maintain emotional balance.",
            "In moments of sadness, it's important to take care of your mental and physical health. Find ways to indulge yourself and interact with loved ones.",
            "You're not alone in this feeling of sadness. If needed, seek professional help to assist you in the healing process and find ways to regain happiness."
        ]
    else:
        alternative_responses = [
            "I understand that you're feeling something unique right now. It may be helpful to reflect on your emotions and seek support from loved ones.",
            "Every emotion has its own significance. Take the time to explore and understand your feelings, and remember that you're not alone in your experiences.",
            "Emotions can be complex, and it's okay to have a mixture of feelings. Be kind to yourself and give yourself space to process and navigate through your emotions."
        ]
    
    alternative_response = random.choice(alternative_responses)
    return alternative_response

# Simulasi chat bot
while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        break
    response = generate_response(user_input)
    alternative_response = generate_alternative_response(user_input)
    print("ChatBot: ", response)
    print("ChatBot (Alternative): ", alternative_response)
