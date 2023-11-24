#!/usr/bin/env python
# coding: utf-8


import numpy as np
# Provided by Alexey on his github repo
import tflite_runtime.interpreter as tflite

# for image processing
from PIL import Image

# These two come with python
from urllib import request
from io import BytesIO

def download_image(url):
    """Downloads the image from URL

    Args:
        url (String): location of image

    Returns:
        Image object: Image as byte array
    """
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    """Convert the image in RGB representation

    Args:
        img (object): object of image
        target_size (Tuple): length and width of target size

    Returns:
        _type_: _description_
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocessor(url, target_size):
    """Downloads an image from given URL, prepares it and then 
    return a ndarray as float32

    Args:
        url (String): location of iamge
        target_size (tuple): Length,Width of images after resizing

    Returns:
        Numpy Array: A numpy array representation of image
    """
    img = download_image(url)
    img = prepare_image(img, target_size)
    # rescale image and convert to numpy array
    x = np.array(img)/255.
    # create a batch with a single image since this is the expected input to the model:
    X = np.array([x], dtype='float32')

    return X


# load the pretrained model

# Thsi is for local use
# interpreter = tflite.Interpreter(model_path='bees-wasps.tflite')
# for Docker use bees-wasps-v2.tflite
interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
interpreter.allocate_tensors()
# get index of input and output tensors
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def predict(url):
    """Provides prediction whether this image is bee or a wasp

    Args:
        url (string): URL where this image is located

    Returns:
        List: A list of predictions for each class
    """
    target_size = (150, 150)
    
    #get image and preprocess it as numpy array
    X = preprocessor(url, target_size)
    
    # Apply model on this image
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    
    # get predictions
    preds = interpreter.get_tensor(output_index)
    
    # get predictions
    float_predictions = preds[0].tolist()
    
    return float_predictions


def lambda_handler(event, context):
    """Helper function to be used with amazon Lambda

    Args:
        event (Dictionary): dictionary containing URL
        context (Unknown): it will be None

    Returns:
        Prediction for iamge: prediction probability
    """
    url = event['url']
    result = predict(url)
    return result