# Multiple Style Transfer
A style transfer algorithm that applies multiple styles to an image.

## How to use
To run the model that the original paper implements, run `python main.py`. To run the different attempts of merging styles, edit and then run `python style_merge.py`. To run the different experiments for photo style transfer, edit and then run `python photo_style.py`. For photo style, you need to [download the NIMA model weights](https://github.com/titu1994/neural-image-assessment/releases) and store them in `NIMA/inception_resnet_weights.h5`.