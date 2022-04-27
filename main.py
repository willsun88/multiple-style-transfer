import tensorflow as tf
from tensorflow.keras.applications import vgg19
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from helpers import import_image, unprocess_image, weighted_loss, VGG_Model
import hyperparameters as hyp
from os import mkdir, path

def train(style_img_path, content_img_path):
    style_img = import_image(style_img_path)
    content_img = import_image(content_img_path)

    model = VGG_Model()
    targets = (model.call(style_img)[0], model.call(content_img)[1])
    content_img = tf.Variable(content_img, dtype=tf.float32)

    unprocessed = unprocess_image(content_img)
    img = Image.fromarray(unprocessed, 'RGB')
    if not path.isdir("results"):
        mkdir("results")
    img.save('results/pre_transfer.png')

    for i in range(hyp.num_epochs):
        with tf.GradientTape() as tape:
            style, content = model(content_img)
            loss = weighted_loss(style, content, targets, hyp.loss_weights)
        grad = tape.gradient(loss, content_img)
        model.optim.apply_gradients([(grad, content_img)])
        content_img.assign(tf.clip_by_value(content_img, 0, 1))

        if (i % hyp.print_every == 0):
            print(f"Epoch {i}")
            unprocessed = unprocess_image(content_img)
            img = Image.fromarray(unprocessed, 'RGB')
            img.save(f'results/transfer_{i}.png')
    
    unprocessed = unprocess_image(content_img)
    img = Image.fromarray(unprocessed, 'RGB')
    img.save('results/post_transfer.png')

if __name__ == "__main__":
    style_img = 'data/udnie.jpg'
    input_img = 'data/chicago.jpg'
    train(style_img, input_img)