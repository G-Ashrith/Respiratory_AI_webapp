import tensorflow as tf

model = tf.keras.models.load_model("model/model.h5", compile=False)

model.save("model/model.keras")