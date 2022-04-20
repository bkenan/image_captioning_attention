# Image Captioning including Attention

The product is designed for  disabled people, who cannot see the pictures but can listen to the description. We are aiming to enable disabled people to access more information and experience the beauty of the world. Our product can predict the captioning of an image and display to users. A user can upload an image and play the audio of the captioning of the image, so that they can "hear" the image.

## Modeling

We have used Image Captioning model in the backend. An Image Captioning model helpt the application to take an image as input and produce a short textual summary describing the content. It uses both Computer Vision and Natural Language Processing to generate the captions.
The model is implementing an Encoder-Decoder architecture. It encodes images to a high-level representation by Convolutional Neural Network (CNN) and then decodes this representation using an NLP algorithm, Recurrent Neural Network(RNN).
In contrast to traditional models, we have also included Attention that helped the model to focus on the most relevant pixels of the image to produce the captions.

## Dataset

[Flickr8k dataset from Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## Application pages

Home page:

![image](https://user-images.githubusercontent.com/97444802/163241489-7ab169b6-2865-4668-be47-a5827a145a47.png)

When you upload an image:

![test](https://user-images.githubusercontent.com/53462948/164121624-ce98a61a-39ab-4956-bdc0-962e9c80b62a.png)


## Demo:


https://user-images.githubusercontent.com/53462948/164121915-549d5cc5-b2de-49f6-9665-62d70cb03427.mov



## Getting started in the local machine:

1. Clone this repo
2. Download [my trained model](https://drive.google.com/file/d/1t3QbSauxSnZhXE1DbuGwiT2AokOsqOjA/view?usp=sharing) and put it in a new "models" folder within the repo directory
4. make install
5. Open Python Shell and run:
    import spacy
    
    from spacy.cli.download import download
    download(model="en_core_web_sm")
6. Run app.py


## Testing the model in Colab:

[![My Colab notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z1sI5wVmoflOggLfIuIIj7qQ0xAICtgn?usp=sharing) 

Download the Image folder from the above dataset, zip the "Image" folder and put it in the current Colab working directory.
