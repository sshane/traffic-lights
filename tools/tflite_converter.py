import tensorflow as tf
import os

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models'))  # todo: ensure this leads to traffic-lights models directory

model_name = "model"

converter = tf.lite.TFLiteConverter.from_keras_model_file("h5_models/"+model_name+".h5")
tflite_model = converter.convert()
open("lite_models/"+model_name+".tflite", "wb").write(tflite_model)
