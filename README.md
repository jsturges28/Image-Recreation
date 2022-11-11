Jacob Sturges, ID 112918980, Fall 2022
# Overview
This is a repository for the DSA 5900 Professional Practice course. To align with the goals of the course, I have developed several models in order to analyze their accuracy against one another. 
# Contents
This repo contains the following:
* **UNET**: A slighty modified UNET model that contains ~469k parameters. [Source](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf)
* **UNET 3+**: A highly modified UNET 3+ model that contains ~4 million parameters. [Source](https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=9053405&ref=aHR0cHM6Ly9pZWVleHBsb3JlLmllZWUub3JnL2Fic3RyYWN0L2RvY3VtZW50LzkwNTM0MDU=&tag=1)
# Objectives
I will be training these models on a modified version of MNIST, a multidigit dataset called the M2NIST dataset. The dataset contains 5000 examples of handwritten digits, in grayscale format, where there may be multiple digits recorded on screen of size 64 x 84. This is translated to a numpy array, represented in binary encodings. The training data is the flattened image of the digits, while the training labeled data is each digit represented by a channel. For example, with a training image containing the images "3" and "4", the digit "3" would be represented alone in channel 4 and the digit "4" would be represented alone in channel 5. So, the training data is named "combined" and has dimension (5000 x 64 x 84 x 1) and the labeled data is named "segmented" and has dimension (5000 x 64 x 84 x 11). The goal of the models are to accurately predict the digits using binary cross-entropy, and recreate them on screen. Source for the dataset can be found here: [https://www.kaggle.com/datasets/farhanhubble/multimnistm2nist].
