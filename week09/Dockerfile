# We'll use a Docker image that was already prepared for us by Alexy. 
# This is the Dockerfile that we used for creating the image:
# FROM public.ecr.aws/lambda/python:3.10
# COPY bees-wasps-v2.tflite .
# And pushed it to agrigorev/zoomcamp-bees-wasps:v2.
# A few notes:
# The image already contains a model and it's not the same model as the one we 
# used for questions 1-4.
# The version of Python is 3.10, so you need to use the right wheel for TF-Lite. 
# For Tensorflow 2.14.0, 
# it's https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

FROM agrigorev/zoomcamp-bees-wasps:v2


# using code from pillow
# RUN pip install keras-image-helper
RUN pip install pillow
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl?raw=true

COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]