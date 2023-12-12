## OpenFlamingo UI Accessibility Research

This repository contains code for exploring multimodal models using OpenFlamingo, an open-source model inspired by DeepMind's Flamingo model. The main goal of this project is to improve the OpenFlamingo model and apply it to research in UI accessibility.

The code includes steps for setting up the environment, downloading necessary datasets, installing required packages, initializing an OpenFlamingo model, downloading pretrained weights, generating text based on images, training the model with specific parameters, evaluating the data, and applying the model to unique UI datasets and their descriptions.

### Prerequisites

Before running the code, make sure you have a Kaggle account. You will need to generate a new token or expire an existing token in your Kaggle settings under the API section. This will automatically download a json file named kaggle.json. 

### Installation

To install the necessary packages, use pip install command as shown in the code. The packages include opendatasets for downloading datasets from Kaggle, open-flamingo for creating and transforming models, pre-commit for managing and maintaining multi-language pre-commit hooks.

### Model Initialization

The OpenFlamingo model is initialized by importing the create_model_and_transforms function from the open_flamingo package. The function creates a model, an image processor, and a tokenizer with specified parameters including paths to vision encoder and language encoder used in the model.

### Training

The train.py script is used to train a multimodal language model using the Multimodal C4 dataset and MPT-1B-RedPajama 200B language model. The script takes several parameters including paths to language model and tokenizer used by language model, number of layers between cross attention layers, batch sizes for different datasets, number of samples to use for training on different datasets among others.

### Evaluation

The evaluation process involves downloading the WordNet database, a large English dictionary used in NLP tasks such as text classification, sentiment analysis, and machine translation. A word cloud is also created from the generated text to display and visualize word frequency.

### Applying Model to UI Dataset and Descriptions

The model is applied to unique UI datasets and their descriptions. The datasets are loaded, preprocessed, and then used as input to the OpenFlamingo model. The generate method of the model is called with the preprocessed images and text as input, along with some additional parameters that control the generation process.

### Comparative Analysis

The cosine similarity between the original and generated descriptions for each image is calculated using TfidfVectorizer from sklearn library.

### Visualizations

Word Cloud is used to visualize word frequency in the generated text. Also, a heatmap is created to visualize cosine similarity matrix.

### Citations

Awadalla, A., Gao, I., Gardner, J., Hessel, J., Hanafy, Y., Zhu, W., Marathe, K., Bitton, Y., Gadre, S., Jitsev, J., Kornblith, S., Koh, P.W., Ilharco, G., Wortsman, M., Schmidt, L. (2023). OpenFlamingo. (Version 2.0.1). https://github.com/mlfoundations/open_flamingo
