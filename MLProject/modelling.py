# modelling.py untuk MLProject - Cyberbullying Detection
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from sklearn.metrics import accuracy_score

# Load preprocessed data
padded_latih = np.load('padded_latih.npy', allow_pickle=True)
padded_test = np.load('padded_test.npy', allow_pickle=True)
labels_latih = np.load('labels_latih.npy', allow_pickle=True)
labels_test = np.load('labels_test.npy', allow_pickle=True)

max_len = padded_latih.shape[1]
num_classes = labels_latih.shape[1]

print(f"Training samples: {len(padded_latih)}")
print(f"Test samples: {len(padded_test)}")
print(f"Max length: {max_len}, Num classes: {num_classes}")

# Enable autolog
# mlflow.tensorflow.autolog()

# with mlflow.start_run():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(max_len,)),
#         tf.keras.layers.Embedding(10000, 16),
#         tf.keras.layers.LSTM(64),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(num_classes, activation='softmax')
#     ])
    
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.00146),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     model.fit(padded_latih, labels_latih, 
#               batch_size=32, 
#               epochs=10, 
#               validation_split=0.1,
#               verbose=1)
    
#     y_pred = model.predict(padded_test)
#     y_pred_classes = np.argmax(y_pred, axis=1)
#     y_test_classes = np.argmax(labels_test, axis=1)
    
#     accuracy = accuracy_score(y_test_classes, y_pred_classes)
#     print(f"Accuracy: {accuracy:.4f}")


mlflow.tensorflow.autolog()

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_len,)),
    tf.keras.layers.Embedding(10000, 16),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00146),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    padded_latih,
    labels_latih,
    batch_size=32,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

y_pred = model.predict(padded_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(labels_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Accuracy: {accuracy:.4f}")