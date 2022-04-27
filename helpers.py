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
    def __init__(self):
        super(VGG_Model, self).__init__()
        self.content_layers = ['block5_conv2'] 
        self.style_layers = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1', 
            'block4_conv1', 
            'block5_conv1'
        ]

        self.vgg = VGG19(include_top=False, weights='imagenet')
        self.vgg.trainable = False
        outp = [self.vgg.get_layer(x).output 
            for x in self.style_layers + self.content_layers]
        self.model = tf.keras.Model([self.vgg.input], outp)

        self.optim = tf.optimizers.Adam(learning_rate=0.01)
    
    def call(self, x):
        # Note: Input expected to be already pre-processed.
        x = vgg19.preprocess_input(x*255.0)
        out = self.model(x)
        # print([x.shape for x in out])
        style, content = out[:len(self.style_layers)], out[len(self.style_layers):]
        style = [gram_mat(layer) for layer in style]
        return (style, content)

def weighted_loss(style, content, targets, loss_weights):
    # Note: assume set_targets has run
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
    """
    copy = img.numpy().copy()*255.0
    if len(copy.shape) == 4:
        copy = np.squeeze(copy, 0)
    return np.clip(copy, 0, 255).astype(np.uint8)
