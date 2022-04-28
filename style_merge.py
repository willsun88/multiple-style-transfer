import tensorflow as tf
from helpers import import_image, unprocess_image, weighted_loss, VGG_Model
import hyperparameters as hyp
from os import mkdir, path
from tqdm import trange

def average_merge_train(style_img_paths, content_img_path):
    """
    Runs the training process using the hyperparameters given in 
    hyperparameters.py, merging multiple styles by averaging their 
    style outputs from the VGG19 model.
    """

    # Creates the result directory, where result images are stored
    if not path.isdir("results"):
        mkdir("results")
    
    # Import and process the images using helper functions
    style_images = []
    for p in style_img_paths:
        style_images.append(import_image(p))
    content_img = import_image(content_img_path)

    # Define the model
    model = VGG_Model(hyp.content_layers, hyp.style_layers)

    # Create the average style
    styles = []
    for starting_style_img in style_images:
        styles.append(model.call(starting_style_img)[0])
    target_style = []
    for i in range(len(styles[0])):
        curr = None
        for starting_style in styles:
            if curr is None:
                curr = starting_style[i]
            else:
                curr += starting_style[i]
        curr /= len(styles)
        target_style.append(curr)
    targets = (target_style, model.call(content_img)[1])

    # Make content a variable for backpropogation
    content_img = tf.Variable(content_img, dtype=tf.float32)

    # Save what our image looks like before iteration
    unprocess_image(content_img).save('results/pre_transfer.png')

    # Training loop (using a progress bar via tqdm)
    t = trange(hyp.num_epochs)
    for i in t:
        with tf.GradientTape() as tape:
            style, content = model(content_img)
            loss = weighted_loss(style, content, targets, hyp.loss_weights)
        grad = tape.gradient(loss, content_img)
        model.optim.apply_gradients([(grad, content_img)])
        content_img.assign(tf.clip_by_value(content_img, 0, 1))
        t.set_postfix(loss=loss.numpy())

        # Every save_every iterations, we save a result image
        if (i % hyp.save_every == 0):
            unprocess_image(content_img).save(f'results/transfer_{i}.png')
    
    # Save the final results
    unprocess_image(content_img).save('results/post_transfer.png')

if __name__ == "__main__":
    # Run our training loop on some images
    style_img = ['data/van_gogh/starry_night.jpg', 'data/van_gogh/rhone.jpg']
    input_img = 'data/labrador.jpg'
    average_merge_train(style_img, input_img)