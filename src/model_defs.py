# src/model_defs.py
import tensorflow as tf
from tensorflow.keras import layers, models
from config import MODEL

def build_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(MODEL["cnn_filters"], MODEL["cnn_kernel"], activation='relu')(inputs)
    x = layers.MaxPooling1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(MODEL["dense_units"], activation='relu')(x)
    x = layers.Dropout(MODEL["dropout"])(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs, name="cnn")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(MODEL["lstm_units"]))(inputs)
    x = layers.Dense(MODEL["dense_units"], activation='relu')(x)
    x = layers.Dropout(MODEL["dropout"])(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs, name="lstm")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_hybrid(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # CNN branch
    c = layers.Conv1D(MODEL["cnn_filters"], MODEL["cnn_kernel"], activation='relu')(inputs)
    c = layers.MaxPooling1D()(c)
    c = layers.Flatten()(c)

    # BiLSTM branch
    l = layers.Bidirectional(layers.LSTM(MODEL["lstm_units"]))(inputs)

    # Merge
    x = layers.concatenate([c, l])
    x = layers.Dense(MODEL["dense_units"], activation='relu')(x)
    x = layers.Dropout(MODEL["dropout"])(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="hybrid")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
