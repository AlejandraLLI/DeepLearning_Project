# Import libraries 
from datetime import datetime
import tensorflow as tf
import numpy as np
import keras
from keras import Model
from keras.utils import get_file, plot_model
from keras.optimizers import SGD
from tensorflow.keras.applications import vgg19, vgg16, resnet50, inception_v3, densenet

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import typing 
import pathlib as Path

#def gram_matrix(x: tf.Tensor) -> tf.Tensor:
def gram_matrix(x):
    """
    Compute the Gram matrix for a tensor.

    A Gram matrix is the result of multiplying a given matrix by its transposed matrix.
    In the context of computer vision, the Gram matrix of an image tensor represents the
    correlation between feature maps of convolutional neural networks and is used in style
    transfer techniques.

    Args:
        x (tf.Tensor): The input tensor. If x is a 3D tensor like an image, it should 
                       have dimensions [height, width, channels].

    Returns:
        tf.Tensor: The Gram matrix of the input tensor.
    """
    # Transpose the tensor: 2nd dim to 1st, 1st to 2nd, 3rd stays the same.
    x = tf.transpose(x, (2, 0, 1))

    # Reshape the tensor into 2D
    # First dim is first dim of x, second dim such that it has the same
    # number of elements in x
    features = tf.reshape(x, (tf.shape(x)[0], -1))

    # calculate matrix multiplication of features and its transpose. 
    gram = tf.matmul(features, tf.transpose(features))

    # Function output
    return gram


#def style_loss(style: tf.Tensor, combination: tf.Tensor, img_nrows: int, 
#                 img_ncols: int, channels: int = 3) -> tf.Tensor:
def style_loss(style_image, combination_image, img_nrows, 
                img_ncols, channels=3):
    """
    Calculate the style loss between a "style" image and a "combination" image.
    The style loss is defined as an adjusted Mean Squared Error (MSE) of the Gram 
    Matrices of the "style" and "combination" images.

    Args:
        style (tf.Tensor): The style image, represented as a tensor.
        combination (tf.Tensor): The combination image, represented as a tensor.

    Returns:
        tf.Tensor: The style loss, represented as a tensor.
    """
    # Calculate gram matrix of style and combined image
    S = gram_matrix(x = style_image)
    C = gram_matrix(x = combination_image)

    # Calculate image size (total no. of pixels)
    size = img_nrows * img_ncols

    # Return adjusted MSE between style and combined image
    return tf.reduce_sum(tf.square(S - C)) * (1.0 / (4.0 * (channels ** 2) * (size ** 2)))


#def content_loss(base: tf.Tensor, combination: tf.Tensor) -> tf.Tensor:
def content_loss(base_image, combination_image):
    """
    Compute the content loss between a "base" image and a "combination" image.
    The content loss is defined as the sum of the squared differences between the pixel values of the "base" and "combination" images.

    Args:
        base (tf.Tensor): The base image, represented as a tensor.
        combination (tf.Tensor): The combination image, represented as a tensor.

    Returns:
        tf.Tensor: The content loss, represented as a tensor.
    """
    return tf.reduce_sum(tf.square(combination_image - base_image))



#def calculate_total_loss(style: tf.Tensor, base: tf.Tensor, combination: tf.Tensor, 
#               style_layers: typing.List[str], content_layers: str, 
#               style_weight: float, content_weight: float,
#               image_nrows: int, image_ncols: int,
#               mod_name: str = 'vgg19') -> float:
def calculate_total_loss(style_image, base_image, combination_image,
                         style_layers, content_layers,
                         style_weight, content_weight,
                         img_nrows, img_ncols, feature_extractor, channels = 3,
                         mod_name = 'vgg19'):
    """
    Computes the total loss for a neural style transfer task.

    Args:
        style (tf.Tensor): The style image tensor.
        base (tf.Tensor): The base (or content) image tensor.
        combination (tf.Tensor): The combination image tensor to be optimized.
        style_layers (list[str]): A list of layer names to use for style extraction.
        content_layers (list[str]): A list of layer names to use for content extraction.
        style_weight (float): The weight for the style loss.
        content_weight (float): The weight for the content loss.
        image_nrows (int): The number of rows in the image.
        image_ncols (int): The number of columns in the image.
        mod_name (str, optional): The name of the model to use. Defaults to 'vgg19'.

    Returns:
        float: The total loss, measuring how well the combination image matches the content of the base image and the style of the style image.
    """    
    # Combine all images in one tensor 
    input_tensor = tf.concat([base_image, style_image, combination_image], axis = 0)

    # Get layers activations
    features = feature_extractor(input_tensor)

    # Initalize the loss function in zero 
    total_loss = tf.zeros(shape=())

    # Extract features of content image
    for layer_name in content_layers:
        
        layer_features = features[layer_name]

        # Get featrues of base and combination image from specified content layers
        base_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]

        # Update total loss with content loss: 
        # total_loss = total_loss + content_weight * content_loss 
        cl = content_loss(base_image = base_features, combination_image = combination_features)
        total_loss = total_loss + content_weight * cl

    # Iterate over each style layer
    for layer_name in style_layers:

        # Extract features for the style image 
        layer_features = features[layer_name]


        # Get features for style and combination image from specified style layers
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]

        # calculate style loss 
        sl = style_loss(style_image = style_features, combination_image = combination_features,
            img_nrows = img_nrows, img_ncols = img_ncols, channels = channels)

        # Update total loss: 
        #total_loss = total_loss + (style_weight/# layers) style_loss
        total_loss += (style_weight/len(style_layers)) * sl
    
    return total_loss



#def calculate_loss_and_grads(style: tf.Tensor, base: tf.Tensor, combination: tf.Tensor,
#                            style_layers: typing.List[str], content_layers: typing.List[str],
#                            style_weight: float, content_weight: float,
#                            image_nrows: int, image_ncols: int,
#                            mod_name: str = 'vgg19') -> typing.Any:
@tf.function
def calculate_loss_and_grads(style_image, base_image, combination_image,
                        style_layers, content_layers,
                        style_weight, content_weight,
                        img_nrows, img_ncols, feature_extractor, channels = 3,
                        mod_name = 'vgg19'): 
    """
    Computes the total loss and its gradients for a neural style transfer task.

    Args:
        style (tf.Tensor): A tensor representing the style reference image.
        base (tf.Tensor): A tensor representing the base (content) image.
        combination (tf.Tensor): A tensor representing the combination image that should match the content of the 
                                 base image and the style of the style image.
        style_layers (list[str]): Names of the layers in the model that are used for style loss.
        content_layers (list[str]): Names of the layers in the model that are used for content loss.
        style_weight (float): Weight for the style loss.
        content_weight (float): Weight for the content loss.
        image_nrows (int): The number of rows in the images.
        image_ncols (int): The number of columns in the images.
        mod_name (str, optional): Name of the model to use. Defaults to 'vgg19'.

    Returns:
        total_loss (float): The total loss for the style transfer.
        grads (tf.Tensor): The gradients of the total loss with respect to the combination image tensor.

    Raises:
        ValueError: If mod_name is not 'vgg19' or 'vgg16'.
    """
    # set context manager for automatic differentiation. 
    with tf.GradientTape() as tape:
        total_loss = calculate_total_loss(style_image = style_image, base_image = base_image,
                                          combination_image = combination_image, 
                                          style_layers = style_layers, content_layers = content_layers,
                                          style_weight = style_weight, content_weight = content_weight, 
                                          img_nrows = img_nrows, img_ncols = img_ncols, feature_extractor = feature_extractor, 
                                          channels = channels, mod_name = mod_name)
        
        # Calculate the gradients of total_loss with tespect to the combination image
        # Note: here total_loss is the target and combination image is the source.
        # Grads indicates how much each pixel needs to change to reduce the total loss. 
        grads = tape.gradient(total_loss, combination_image)
    
    return total_loss, grads


#def preprocess_image(image_path: Path, img_nrows: int, img_ncols: int, mod_name: str) -> tf.Tensor:
def preprocess_image(image_path, img_nrows, img_ncols, mod_name):
    """
    Preprocesses an image by loading it, resizing it, and converting it into a tensor.

    Args:
        image_path (Path): The path to the image.
        img_nrows (int): The number of rows (height) to resize the image to.
        img_ncols (int): The number of columns (width) to resize the image to.
        mod_name (str): The name of the model used for preprocessing. Currently supports 
                        'vgg19' and 'vgg16', 'resnet50', and 'densenet121'

    Returns:
        tf.Tensor: The preprocessed image as a tensor.

    Raises:
        NotImplementedError: If the specified model name is not supported.
    """
    # Use keras preprocessing function to open, resize and format images into 
    # appropriate tensors
    img = tf.keras.utils.load_img(image_path, 
                                             target_size=(img_nrows, img_ncols))

    # Convert preprocessed image to a numpy array
    img = tf.keras.utils.img_to_array(img)

    # Add an extra dimension to the array at axis 0. 
    # Note: this is to satisfy the infput shape of (batch_size, height, width, channels)
    # of the input tensors. 
    img = np.expand_dims(img, axis=0)

    # Perform preprocessing operations specific to the selected model.
    if mod_name == 'vgg19':
        img = vgg19.preprocess_input(img)
    elif mod_name == 'vgg16':
        img = vgg16.preprocess_input(img)
    elif mod_name == 'resnet50':
        img = resnet50.preprocess_input(img)
    elif mod_name == 'densenet121':
        img = densenet.preprocess_input(img)
    else:
        raise NotImplementedError(f"Model '{mod_name}' not supported.")
    
    # Return image as a tensor. 
    return tf.convert_to_tensor(img)


#def deprocess_image(x: tf.Tensor, img_nrows: int, img_ncols: int):
def deprocess_image(x, img_nrows, img_ncols):
    """
    Converts a tensor into an image by reshaping it and reverting preprocessing steps.

    Args:
        x (tf.Tensor): The input tensor to be converted into an image.
        img_nrows (int): The number of rows the output image should have.
        img_ncols (int): The number of columns the output image should have.

    Returns:
        np.array: The deprocessed image.

    Note: 
        This function specifically undoes the preprocessing steps performed by vgg19.preprocess_input.
        For other models, the specific preprocessing steps would need to be undone differently.
    """

    # Conver array into tensor
    x = x.reshape((img_nrows, img_ncols, 3))

    # Undo zero mean transforamtion: add average for each channel. 
    # The mean values for ImageNet are [103.939, 116.779, 123.68] for BGR channels
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # Convert from BGR to RGB.
    x = x[:, :, ::-1]

    # Make sure it is between 0 y 255
    x = np.clip(x, 0, 255).astype("uint8")

    return x


#def train_StyleTransfer(style_path: Path, base_path: Path, n_iter,
#                        style_layers: typing.List[str], content_layers: typing.List[str],
#                        style_weight: float, content_weight: float, final_img_path: Path, 
#                        mod_name: str = 'vgg19', learning_rate: float = 100,
#                        decay_steps: int = 100, decay_rate = 0.96) -> None:
def train_StyleTransfer(style_path, base_path, n_iter,
                        style_layers, content_layers,
                        style_weight, content_weight, final_img_path,
                        feature_extractor, mod_name = 'vgg19', learning_rate = 100,
                        decay_steps = 100, decay_rate = 0.96, channels = 3):

    # Get size of image
    width, height = tf.keras.utils.load_img(base_path).size

    # Adjust height to 400 and corresponding width to maintain proportions.
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    # Define stochastic gradient descent (sgd) with exponential decay as optimizer for training
    optimizer = SGD(
        tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = learning_rate,
                                                    decay_steps=decay_steps, 
                                                    decay_rate= decay_rate))

    # Preprocess base and style images
    base_image = preprocess_image(image_path = base_path, img_nrows = img_nrows, 
                                  img_ncols = img_ncols, mod_name = mod_name)
    style_image = preprocess_image(image_path = style_path, img_nrows = img_nrows,
                                   img_ncols = img_ncols, mod_name = mod_name)

    # Set combined image as base image
    combination_image = tf.Variable(preprocess_image(image_path = base_path, img_nrows = img_nrows, 
                                  img_ncols = img_ncols, mod_name = mod_name))

    # For each iteration
    for i in range(1, n_iter + 1):
        # Compute total loss and gradients f
        total_loss, grads = calculate_loss_and_grads(style_image = style_image, base_image = base_image,
                                                     combination_image = combination_image,
                                                     style_layers = style_layers, content_layers = content_layers,
                                                     style_weight = style_weight, content_weight = content_weight,
                                                     img_nrows = img_nrows, img_ncols = img_ncols, feature_extractor = feature_extractor, 
                                                     channels = channels, mod_name = mod_name)

        # Apply gradients to the combination image to update it
        optimizer.apply_gradients([(grads, combination_image)])

        if i % 100 == 0:
            print("Iteration %d: loss=%.2f" % (i, total_loss))

    # Deprocess final image 
    img = deprocess_image(x = combination_image.numpy(), img_nrows = img_nrows, img_ncols = img_ncols)

    # Save image to path 
    tf.keras.utils.save_img(final_img_path, img)

    return img