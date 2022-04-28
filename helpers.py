import tensorflow as tf
from tensorflow.keras.applications import vgg19, VGG19
from PIL import Image
import numpy as np
import hyperparameters as hyp

def gram_mat(x):
    """
    Calculates the gram matrix value for a given input tensor.
    """
    res = tf.linalg.einsum('ijkl,ijkm->ilm', x, x)
    return res / (tf.cast(x.shape[1]*x.shape[2], tf.float32))

class VGG_Model(tf.keras.models.Model):
    """
    Class to run the VGG model on style/content images.
    """
    def __init__(self, content_layers, style_layers):
        super(VGG_Model, self).__init__()
        # Define the blocks we will use to represent content and style.
        self.content_layers = content_layers
        self.style_layers = style_layers

        # Initialize a VGG19 model that will return the layers we want.
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outp = [vgg.get_layer(x).output 
            for x in self.style_layers + self.content_layers]
        self.model = tf.keras.Model([vgg.input], outp)

        # Initialize our optimizer.
        self.optim = tf.optimizers.Adam(learning_rate=0.01)
    
    def call(self, x):
        # Process the input and run it through the model.
        # Note: Input expected to be already pre-processed.
        x = vgg19.preprocess_input(x*255.0)
        out = self.model(x)

        # Split content and style, and run style layers through the gram matrix
        # function.
        style, content = out[:len(self.style_layers)], out[len(self.style_layers):]
        style = [gram_mat(layer) for layer in style]
        return (style, content)

def weighted_loss(style, content, targets, loss_weights):
    """
    Calculates a loss weighted between style reconstruction loss and 
    content reconstruction loss. 
    params:
        style - a tensor representing the style outputs 
        content - a tensor representing the content outputs
        targets - a tuple where the first element is the target style output 
            (the style output of the style image) and the second element is 
            the target content output (the content output of the content image)
        loss_weights - a tuple where the first element is the weight given to 
            the style reconstruction loss and the second element is the weight 
            given to the content reconstruction loss.
    output: a float representing the weighted loss
    """
    style_losses = [tf.reduce_mean((style[i]-targets[0][i])**2) 
                        for i in range(len(style))]
    style_loss = tf.reduce_sum(style_losses)
    style_loss = style_loss * loss_weights[0] / len(style)

    content_losses = [tf.reduce_mean((content[i]-targets[1][i])**2) 
                        for i in range(len(content))]
    content_loss = tf.reduce_sum(content_losses)
    content_loss = content_loss * loss_weights[1] / len(content)
    
    return style_loss + content_loss

def import_image(file_path):
    """
    Imports an image from a file path and pre-processes it for VGG.
    params:
        file_path - a string representing the relative file path of the image
    output: a tensor representing the image, processed into the shape needed 
        for the call to an instance of VGG_Model.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, 3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    cut_shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale_factor = hyp.max_dim / max(cut_shape)

    img = tf.image.resize(img, tf.cast(cut_shape * scale_factor, tf.int32))
    return tf.expand_dims(img, axis=0)

def unprocess_image(img):
    """
    Reverses formatted image processing for display.
    params:
        img - a tensor representing an image that has been processed by 
            import_image
    output: a PIL image instance of the image 
    """
    copy = img.numpy().copy()*255.0
    if len(copy.shape) == 4:
        copy = np.squeeze(copy, 0)
    unprocessed = np.clip(copy, 0, 255).astype(np.uint8)
    return Image.fromarray(unprocessed, 'RGB')
