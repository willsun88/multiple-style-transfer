import tensorflow as tf
from helpers import import_image, unprocess_image, weighted_loss, VGG_Model
import hyperparameters as hyp
from os import mkdir, path
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

def train(style_img_path, content_img_path, loss_graph=False):
    """
    Runs the training process using the hyperparameters given in 
    hyperparameters.py
    """

    # Creates the result directory, where result images are stored
    if not path.isdir("outputs"):
        mkdir("outputs")
    
    # Import and process the images using helper functions
    style_img = import_image(style_img_path)
    content_img = import_image(content_img_path)

    # Define the model and the targets for our style and content, and
    # make content a variable for backpropogation
    model = VGG_Model(hyp.content_layers, hyp.style_layers)
    targets = (model.call(style_img)[0], model.call(content_img)[1])
    content_img = tf.Variable(content_img, dtype=tf.float32)

    # Save what our image looks like before iteration
    unprocess_image(content_img).save('outputs/pre_transfer.png')

    # Training loop (using a progress bar via tqdm)
    losses = []
    t = trange(hyp.num_epochs)
    for i in t:
        with tf.GradientTape() as tape:
            style, content = model(content_img)
            loss = weighted_loss(style, content, targets, 
                hyp.loss_weights, content_img)
        
        if loss_graph:
                losses.append(loss)

        grad = tape.gradient(loss, content_img)
        model.optim.apply_gradients([(grad, content_img)])
        content_img.assign(tf.clip_by_value(content_img, 0, 1))
        t.set_postfix(loss=loss.numpy())

        # Every save_every iterations, we save a result image
        if (i % hyp.save_every == 0):
            unprocess_image(content_img).save(f'outputs/transfer_{i}.png')
    
    # Graph losses if necessary
    if loss_graph:
        plt.plot(np.arange(len(losses)), losses)
        plt.ylabel('Loss')
        plt.xlabel('Training Step')
        plt.title('Loss Curve For Single Style Transfer')
        plt.show()

        plt.plot(np.arange(len(losses)), np.log(np.array(losses)))
        plt.ylabel('Log Loss')
        plt.xlabel('Training Step')
        plt.title('Log Loss Curve For Single Style Transfer')
        plt.show()

    # Save the final results
    unprocess_image(content_img).save('outputs/post_transfer.png')

if __name__ == "__main__":
    # Run our training loop on some images
    style_img = 'data/van_gogh/starry_night.jpg'
    input_img = 'data/labrador.jpg'
    train(style_img, input_img, loss_graph=True)