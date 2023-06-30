# image-captioning


Setup and Download the dataset: Necessary modules are imported. These modules include numpy, tensorflow, keras, and its layers. In addition, efficientnet and TextVectorization from tensorflow.keras.applications and tensorflow.keras.layers are imported respectively for efficient image processing and text vectorization. Seed for reproducibility is also set. The Flickr8K dataset, which consists of over 8,000 images, each paired with five different captions, is downloaded and unzipped.

Define constants: Various constants for the model and training are defined. These include image size, vocabulary size, sequence length, image and token embedding dimensions, per-layer units in the feed-forward network, batch size, epochs, and AUTOTUNE for optimizing data loading.

Prepare the Dataset: A function load_captions_data(filename) is defined to map each image with its corresponding caption. The function reads the caption file, maps each image (key) with its corresponding captions (values), and returns a dictionary with the mapping and a list of all the available captions. The function train_val_split(caption_data, train_size=0.8, shuffle=True) is used to split the loaded caption data into training and validation sets.

Vectorizing the text data: The TextVectorization layer is used to vectorize the text data. It turns the original strings into integer sequences where each integer represents the index of a word in a vocabulary. Custom string standardization is implemented to strip punctuation characters except < and >, and then the default splitting scheme (split on whitespace) is used. After that, the vectorization layer is adapted to the text data.

Data augmentation for image data: A Sequential model is created for image data augmentation. This includes horizontal flipping, rotation, and contrast alteration.

Building a tf.data.Dataset pipeline for training: This step involves creating a tf.data.Dataset object for pairs of images and corresponding captions. The function decode_and_resize(img_path, size=IMAGE_SIZE) is defined to read and resize an image. The function read_train_image(img_path, size=IMAGE_SIZE) is defined to read and augment the image data for training. The function read_valid_image(img_path, size=IMAGE_SIZE) is used to read the image data for validation. The function make_dataset(images, captions, split="train") is used to create a dataset of image-caption pairs, tokenize the captions, and batch the dataset for training or validation.

The resulting train_dataset and valid_dataset are now ready for model training. The model will take in the image and caption pairs from these datasets and learn to generate captions for the images.

CNN model: It starts by defining a function get_cnn_model that sets up a convolutional neural network (CNN) using the EfficientNetB0 model. This CNN will be used to extract features from images.

TransformerEncoderBlock: This class sets up a transformer encoder block. It contains a MultiHeadAttention layer and LayerNormalization layers. The call function normalizes the inputs, applies a dense layer, and then applies multi-head attention.

PositionalEmbedding: This class is used to add positional embeddings to the input sequences. It contains an embedding layer for the tokens and one for the positions.

TransformerDecoderBlock: This class sets up a transformer decoder block. It contains two MultiHeadAttention layers, a feed forward network (consisting of two dense layers), and LayerNormalization layers. The call function applies embeddings to the inputs, calculates masks, applies attention and the feed forward network, and returns the predictions.

ImageCaptioningModel: This is the main model class. It contains a method to calculate the loss, a method to calculate the accuracy, and train/test step methods for training and testing the model. The train_step method gets image embeddings, passes them and each of five captions to the decoder to compute loss and accuracy, applies gradients, and updates the trackers. The test_step method is similar, but does not apply gradients.

Model training: The script then gets the CNN model and constructs the transformer encoder and decoder blocks. It sets up the image captioning model with these components. It defines a sparse categorical cross-entropy loss function and early stopping as the callback. It defines a learning rate schedule and compiles the model with these settings. It then trains the model with the train_dataset.

Sample predictions: After training, the script generates some captions for random images in the validation set. It uses the trained model to generate these captions.

The output of this code would be the model that has been trained to generate captions for images. You could feed any image to this model, and it will generate a text description (or "caption") of the image. The quality of the captions will depend on how well the model has been trained.

Note: The last line while True: pass is used to keep the script running indefinitely in a Colab environment. It can be removed if you are running the script in a different environment.
