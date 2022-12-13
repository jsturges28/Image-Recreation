Jacob Sturges, ID 112918980, Fall 2022
# Overview
This is a repository for the DSA 5900 Professional Practice course. To align with the goals of the course, I have developed several models in order to analyze their accuracy against one another. 
# Contents
This repo contains the following:
* **UNET**: A slighty modified UNET model that contains ~469k parameters. [Source](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf)
* **UNET 3+**: A highly modified UNET 3+ model that contains ~4 million parameters. [Source](https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=9053405&ref=aHR0cHM6Ly9pZWVleHBsb3JlLmllZWUub3JnL2Fic3RyYWN0L2RvY3VtZW50LzkwNTM0MDU=&tag=1)
# Objectives
I will be training these models on a modified version of MNIST, a multidigit dataset called the M2NIST dataset. The dataset contains 5000 examples of handwritten digits, in grayscale format, where there may be multiple digits recorded on screen of size 64 x 84. This is translated to a numpy array, represented in binary encodings. The training data is the flattened image of the digits, while the training labeled data is each digit represented by a channel. For example, with a training image containing the images "3" and "4", the digit "3" would be represented alone in channel 4 and the digit "4" would be represented alone in channel 5. So, the training data is named "combined" and has dimension (5000 x 64 x 84 x 1) and the labeled data is named "segmented" and has dimension (5000 x 64 x 84 x 11). The goal of the models are to accurately predict the digits using binary cross-entropy, and recreate them on screen. Source for the dataset can be found [here](https://www.kaggle.com/datasets/farhanhubble/multimnistm2nist).
# Instructions
- Donwload the combined and segmented datasets [here](https://www.kaggle.com/datasets/farhanhubble/multimnistm2nist?resource=download&select=segmented.npy). Extract them into the appropriate repository.
- If you want to run through your terminal, you're given a list of arguments that you may invoke to help train your model:
  - **--epochs**: Specify number of training epochs. Default: 20.
  - **--val_acc**: Specify max accuracy before model stops training (Binary accuracy only). Default: 0.95.
  - **--min_delta**: Specify minimum training accuracy improvement for each epoch. Default: 0.005.
  - **--batch_size**: Specify size of batch per epoch. Default: 50.
  - **--no_display**: Skip displaying set of learning curve(s)/don't save figures after running experiment(s). Default: False.
  - **--no_results**: Skip predicting values and dont display the handwritten digits. Default: False.
  - **--no_verbose**: Skip the training progress display and don't print results to screen. Default: False.
  ### TODO:
  - [ ] Add argparse functionality for toggling analytics.
- You may also wish to edit and run the script.sh file I have provided, which is a simple script that runs the model of your choosing for n amount of times.
  - To edit the script file, download and install [nano text editor](https://www.nano-editor.org/download.php) and invoke ```nano script.sh``` in your terminal to edit the file. Change the higher number in the for loop to the desired amount of experiments to run.
- The program will run the model and save the metrics into a .pkl file for each experiment, as well as plotting a figure comparing all of the previous runs by IOU-score, which is saved in a 'figures' folder. It will automatically create a 'results' folder that will hold the pickled files and a 'figures' folder that will hold the charts. The experimental index is automatically incremented to ensure uniqueness and ease finding the files you want.
- The analytics.py file contains a few ways to extract average and maximum metrics for each model, as well as plotting the best models against one another in a single chart. You may choose to run this in your terminal as a standalone file.
  ### TODO:
  - [ ] add more analytics functionality.
