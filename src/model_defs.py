# small, fast models compatible with SEQUENCE_LENGTH==1
import tensorflow as tf
from tensorflow.keras import layers, models
from config import MODEL


def build_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(MODEL['cnn_filters'], MODEL['cnn_kernel'], activation='relu')(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(MODEL['dense_units'], activation='relu')(x)
    x = layers.Dropout(MODEL['dropout'])(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    m = models.Model(inputs, outputs)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def build_lstm(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(MODEL['lstm_units']))(inputs)
    x = layers.Dense(MODEL['dense_units'], activation='relu')(x)
    x = layers.Dropout(MODEL['dropout'])(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    m = models.Model(inputs, outputs)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def build_hybrid(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    c = layers.Conv1D(MODEL['cnn_filters'], MODEL['cnn_kernel'], activation='relu')(inputs)
    c = layers.Flatten()(c)
    l = layers.Bidirectional(layers.LSTM(MODEL['lstm_units']))(inputs)
    x = layers.concatenate([c, l])
    x = layers.Dense(MODEL['dense_units'], activation='relu')(x)
    x = layers.Dropout(MODEL['dropout'])(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    m = models.Model(inputs, outputs)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m