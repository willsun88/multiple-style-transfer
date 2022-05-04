import tensorflow as tf
from helpers import import_image, unprocess_image, weighted_loss, multiple_weighted_loss, VGG_Model
import hyperparameters as hyp
from os import mkdir, path
from tqdm import trange
import numpy as np

def average_merge_train(style_img_paths, content_img_path):
    """
    Runs the training process using the hyperparameters given in 
    hyperparameters.py, merging multiple styles by averaging their 
    style outputs from the VGG19 model.
    """

    # Creates the result directory, where result images are stored
    if not path.isdir("outputs"):
        mkdir("outputs")
    
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
    unprocess_image(content_img).save('outputs/pre_transfer.png')

    # Training loop (using a progress bar via tqdm)
    t = trange(hyp.num_epochs)
    for i in t:
        with tf.GradientTape() as tape:
            style, content = model(content_img)
            loss = weighted_loss(style, content, targets, hyp.loss_weights,
                content_img)
        grad = tape.gradient(loss, content_img)
        model.optim.apply_gradients([(grad, content_img)])
        content_img.assign(tf.clip_by_value(content_img, 0, 1))
        t.set_postfix(loss=loss.numpy())

        # Every save_every iterations, we save a result image
        if (i % hyp.save_every == 0):
            unprocess_image(content_img).save(f'outputs/transfer_{i}.png')
    
    # Save the final results
    unprocess_image(content_img).save('outputs/post_transfer.png')

def multiple_loss_merge_train(style_img_paths, content_img_path):
    """
    Runs the training process using the hyperparameters given in 
    hyperparameters.py, merging multiple styles by getting the style
    loss for each image in the loss function.
    """

    # Creates the result directory, where result images are stored
    if not path.isdir("outputs"):
        mkdir("outputs")
    
    # Import and process the images using helper functions
    style_images = []
    for p in style_img_paths:
        style_images.append(import_image(p))
    content_img = import_image(content_img_path)

    # Define the model
    model = VGG_Model(hyp.content_layers, hyp.style_layers)

    # Create all the style targets, as well as the content target
    target_style = []
    for starting_style_img in style_images:
        target_style.append(model.call(starting_style_img)[0])
    targets = (target_style, model.call(content_img)[1])

    # Make content a variable for backpropogation
    content_img = tf.Variable(content_img, dtype=tf.float32)

    # Save what our image looks like before iteration
    unprocess_image(content_img).save('outputs/pre_transfer.png')

    # Training loop (using a progress bar via tqdm)
    t = trange(hyp.num_epochs)
    for i in t:
        with tf.GradientTape() as tape:
            style, content = model(content_img)
            loss = multiple_weighted_loss(style, content, targets, 
                hyp.loss_weights, content_img)
        grad = tape.gradient(loss, content_img)
        model.optim.apply_gradients([(grad, content_img)])
        content_img.assign(tf.clip_by_value(content_img, 0, 1))
        t.set_postfix(loss=loss.numpy())

        # Every save_every iterations, we save a result image
        if (i % hyp.save_every == 0):
            unprocess_image(content_img).save(f'outputs/transfer_{i}.png')
    
    # Save the final results
    unprocess_image(content_img).save('outputs/post_transfer.png')

def cut_image_merge_train(style_img_paths, content_img_path, num_splits=5):
    """
    Runs the training process using the hyperparameters given in 
    hyperparameters.py, merging multiple styles by splitting the image
    we optimize into slices.

    Note: Limited to 2 styles.
    """

    # Creates the result directory, where result images are stored
    if not path.isdir("outputs"):
        mkdir("outputs")
    
    # Import and process the images using helper functions
    style_images = []
    for p in style_img_paths:
        style_images.append(import_image(p))
    content_img = import_image(content_img_path)

    # Define the model
    model = VGG_Model(hyp.content_layers, hyp.style_layers)

    # Split the image, define gradient
    splits = np.array_split(content_img, 5, axis=1)
    gradients = np.linspace(0, 1, num=num_splits)

    split_results = []
    # Run on splits
    for i in range(num_splits):
        # Split the image into the appropriate split
        curr_content_split = splits[i]
        style1_grad = gradients[i]
        style2_grad = 1 - gradients[i]

        # Create the average style
        styles = []
        for starting_style_img in style_images:
            styles.append(model.call(starting_style_img)[0])
        target_style = []
        for i in range(len(styles[0])):
            curr = styles[0][i] * style1_grad + styles[1][i] * style2_grad
            target_style.append(curr)
        targets = (target_style, model.call(curr_content_split)[1])

        # Make content a variable for backpropogation
        curr_content_split = tf.Variable(curr_content_split, dtype=tf.float32)

        # Training loop (using a progress bar via tqdm)
        t = trange(hyp.num_epochs)
        for i in t:
            with tf.GradientTape() as tape:
                style, content = model(curr_content_split)
                loss = weighted_loss(style, content, targets, hyp.loss_weights,
                    curr_content_split)
            grad = tape.gradient(loss, curr_content_split)
            model.optim.apply_gradients([(grad, curr_content_split)])
            curr_content_split.assign(tf.clip_by_value(curr_content_split, 0, 1))
            t.set_postfix(loss=loss.numpy())
        
        # Append to result list
        split_results.append(curr_content_split)
    
    # Combine the results
    content_img = tf.Variable(np.concatenate(split_results, axis=1))

    # Save the final results
    unprocess_image(content_img).save('outputs/post_transfer.png')


if __name__ == "__main__":
    # Run our training loop on some images
    # style_img = [
    #     'data/van_gogh/starry_night.jpg', 
    #     'data/van_gogh/rhone.jpg',
    #     'data/van_gogh/field.jpg',
    #     'data/van_gogh/orchard.jpg',
    #     'data/van_gogh/seascape.jpg',
    #     'data/van_gogh/wheat.jpg',
    # ]
    # style_img = [
    #     'data/kandinsky.jpg',
    #     'data/monet.jpg',
    #     'data/udnie.jpg'
    # ]
    # style_img = [
    #     'data/time_of_day/nighttime.jpg', 
    #     'data/time_of_day/nighttime_1.jpg',
    #     'data/time_of_day/nighttime_2.jpg',
    #     'data/time_of_day/nighttime_3.jpg',
    #     'data/time_of_day/nighttime_4.jpg',
    #     'data/time_of_day/nighttime_5.jpg',
    #     'data/time_of_day/nighttime_6.jpg',
    #     'data/time_of_day/nighttime_7.jpg',
    #     'data/time_of_day/nighttime_8.jpg',
    #     'data/time_of_day/nighttime_9.jpg',
    # ]
    style_img = [
        'data/van_gogh/starry_night.jpg', 
        'data/van_gogh/rhone.jpg',
    ]
    input_img = 'data/labrador.jpg'
    # average_merge_train(style_img, input_img)
    # multiple_loss_merge_train(style_img, input_img)
    cut_image_merge_train(style_img, input_img)