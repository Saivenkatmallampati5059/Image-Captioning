import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from netvlad_layer import NetVLAD
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

# Load Kaggle API credentials (replace with your own credentials)
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_api_key'

# Download and unzip the dataset from Kaggle
!kaggle datasets download dataset_name
!unzip dataset_name.zip

# Load image data and caption data from Kaggle dataset
image_data = np.load('path_to_image_data.npy')  # Replace with actual path
caption_data = np.load('path_to_caption_data.npy')  # Replace with actual path

# Preprocess caption data
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(caption_data)
vocab_size = len(tokenizer.word_index) + 1

caption_sequences = tokenizer.texts_to_sequences(caption_data)
max_caption_length = max(len(seq) for seq in caption_sequences)
padded_caption_sequences = pad_sequences(caption_sequences, maxlen=max_caption_length, padding='post')

# Split data into training and validation sets
image_train, image_val, caption_train, caption_val = train_test_split(image_data, padded_caption_sequences, test_size=0.2, random_state=42)

# Load pre-trained ResNet-50 model (excluding the classification layers)
resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
encoder_output = resnet_model.layers[-1].output  # Output from the last convolutional layer

# Add NetVLAD layer for feature aggregation
netvlad_layer = NetVLAD(k_centers=64)(encoder_output)

# Define LSTM decoder
decoder_input = Input(shape=(None,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_input)
lstm_layer = LSTM(units=hidden_units, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(rate=0.5)(lstm_layer)
output_layer = Dense(units=vocab_size, activation='softmax')(dropout_layer)

decoder_model = Model(inputs=decoder_input, outputs=output_layer)

# Combine the encoder and decoder
image_input = Input(shape=(224, 224, 3))
encoded_image = resnet_model(image_input)
netvlad_features = netvlad_layer(encoded_image)
caption_output = decoder_model(netvlad_features)

image_caption_model = Model(inputs=image_input, outputs=caption_output)

# Compile the model
image_caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
image_caption_model.fit(image_train, caption_train, validation_data=(image_val, caption_val), epochs=num_epochs, batch_size=batch_size)
# Generate captions for validation images
def generate_captions(model, tokenizer, image_data):
    captions = []
    for image in image_data:
        netvlad_features = netvlad_layer.predict(np.expand_dims(image, axis=0))
        generated_caption = generate_caption(model, netvlad_features, tokenizer)
        captions.append(generated_caption)
    return captions

def generate_caption(model, features, tokenizer):
    initial_state = None  # You can initialize the LSTM state here if needed
    generated_caption = []
    
    for _ in range(max_caption_length):
        caption_word_probs = model.predict(initial_state=initial_state, steps=1)
        caption_word_idx = np.argmax(caption_word_probs)
        generated_caption.append(caption_word_idx)
        
        if caption_word_idx == end_token_idx:
            break
    
    generated_caption_words = tokenizer.sequences_to_texts([generated_caption])[0].split()
    return generated_caption_words

# Generate captions for validation images
val_captions = generate_captions(decoder_model, tokenizer, image_val)

# Convert validation captions to reference format
val_references = [[caption.split()] for caption in val_captions]

# Convert model-generated captions to hypothesis format
val_hypotheses = val_captions

# Calculate BLEU scores
bleu1_score = corpus_bleu(val_references, val_hypotheses, weights=(1, 0, 0, 0))
bleu2_score = corpus_bleu(val_references, val_hypotheses, weights=(0.5, 0.5, 0, 0))
bleu3_score = corpus_bleu(val_references, val_hypotheses, weights=(0.33, 0.33, 0.33, 0))
bleu4_score = corpus_bleu(val_references, val_hypotheses)

print("BLEU-1 Score:", bleu1_score)
print("BLEU-2 Score:", bleu2_score)
print("BLEU-3 Score:", bleu3_score)
print("BLEU-4 Score:", bleu4_score)















