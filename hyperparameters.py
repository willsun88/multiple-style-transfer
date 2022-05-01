# Maximum dimension for our images
max_dim = 512

# The weights for our losses (style, content, TV)
# loss_weights = (1e-2, 1e4, 1)
# loss_weights = (1e4, 1, 1)
# loss_weights = (1e2, 1, 1)
loss_weights = (1, 1, 1)

# The weights for our losses in the photo style
# transfer case (style, content, NIMA)
photo_loss_weights = (1e2, 1, 1e5)

# Number of epochs run
num_epochs = 1000

# Number of iterations until we save
save_every = 100

# Model style layers
# Taken from the paper's choice of blocks from VGG19.
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1', 
    'block4_conv1', 
    'block5_conv1'
]
# style_layers = [
#     'block1_conv1', 
#     'block2_conv1', 
#     'block3_conv1'
# ]

# Model content layers
# Taken from the paper's choice of blocks from VGG19.
# content_layers = ['block5_conv2'] 
# content_layers = ['block2_conv1']
# content_layers = ['block3_conv2']
# content_layers = ['block1_conv1']
content_layers = ['block4_conv2']