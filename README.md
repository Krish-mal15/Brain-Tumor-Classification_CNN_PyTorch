# Brain-Tumor-Classification_CNN_PyTorch


## Overview:
This goal of this project is to be able to identify if a patient has a brain tumor present and the type of brain tumor it is. This deep learning model can hopefully assist in improving accuracy and efficiency of diagnosis' in the neuroscience field. Brain tumors are one of the most deadly and serious diseases out there so with the use of articial intelligence as such, the menacing disease can be easily identified, diagnosed, and cured. A method to identify brain tumors is to take a MRI (Magnetic Resonance Image) and look for abnormalities. This step can be improved and automated using deep learning models like this one. In fact, using artificial intelligence, along with a more accurate diagnosis, the model can identify early stages, which is very difficult to see, even for expert radiologists and even classify the type of brain tumor. This model is a Convolutional Neural Network with two fully connected layers and three convolutional layers in each. This had the best results of the model architectures I tested. I used PyTorch to build, compile, test, and implement the model. The model yields about a 95% accuracy and works on almost any MRI imaging data due to the versatility of the dataset. Lastly, one should note that if running model on a cpu like I am, for complex model architectures like I have chose, avoid using Keras as it took 4 hours to train and only yielded a 25% accuracy.


## Data:
 - Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
 - __Identifies if__:

   1. **No** Brain Tumor Present
   2. **Glioma** Tumor
   3. **Pituitary** Tumor
   4. **Meningioma** TUmor
 
  - The dataset includes MRI images from different angles of the head, enabling one to see if a tumor on the brain is present
  - Images were mainly 512x512 but there were variants so I resized all images to 224x224
  - Applied a mild amount of data augmentation to challenge the model for best results which included normalizing the image, changing sizes, and, of course, converting the data to tensors


## Main Model Architecture:
I developed the neural network in PyTorch as it's one of the most versatile and popular frameworks. I also really like how you can control all features like the training loop and learning rate unlike other frameworks that provide everything for you. This convolutional neural network is made of two fully connected layers and in each, it has three convolutional layers. Three fully connected layers resulted in overfitting, and the three resulted conv. layers resulted in better accuracis than two without overfitting. It follows a very standard format with in each of the two layers, I apply a covolutional layer with an appropriate amount of neurons followed by a ReLU (Rectified Linear Unit) function and then finally a Max Pooling Layer (with obviously all 2D because a 2D image). The input shape of the model is [32, 224, 224, 3]. The first element is the batch size, second and third the image size, and the fourth is the number of input channels. Furthermore, due to issues in geting the correct output size, I had to make a new variable to calculate the output shape based on the number of Max Pooling layers and hidden neurons. Moreover, I made a forward pass as well as a flatten and linear layer to convert the output into an interperatable vector. Additionally I chose the number of hidden units based on the number of classes I had which was four. Below is a diagram I developed to provide a visual aid to the neural network I developed.

    
![Brain Tumor Classifier Deep Learning Model (CNN) - Main Architecture - Krishna Malhotra](https://github.com/Krish-mal15/Brain-Tumor-Classification_CNN_PyTorch/assets/156712020/13137a04-134d-438d-87a3-93a3cf0b7398)


## Model Losses, Optimizers, Activation and details:
Diving into more specific details of the model, I applied a ReLU activation function after every convolutional layer to prevent negative values and introduce non-linearity to the model for more complex learning capabilities. Furthermore, after extensive testing, I found the Adamax optimizer to be the most efficient. The model ran relatively fast and yielded great results. In simple terms, the Adamax optimizer acccounts for changes (in the squared gradient) to adjust learning rates for model efficiency. Moreover, I used the categorical-cross-entropy loss function as that is the best for classification tasks. Essentially, it uses a softmax function as well as it's own to convert raw values to probabilities of classes which can be converted from the tensor format to something interperatable. Lastly, going back to the main model architecture, it is standard to "Flatten" the output of thre model before coverting it into probabilities as it converts the output to a normalized vector that can be converted easily. You can see this in the above diagram on the top right. Below is a vizualization of the model details. 


![Brain  Tumor Classifier Deep Learning Model (CNN) - Optimizers, Losses, Activation - Krishna Malhotra (1)](https://github.com/Krish-mal15/Brain-Tumor-Classification_CNN_PyTorch/assets/156712020/fd01bbf3-c484-46d2-a5f1-51e911aeafb1)


## Implementation:
In the main.py file, I applied the same image transofrmations to a test image. Then, I used the torch function to decode the tensor output. Using the torch.argmax function, it returns the class with the highest probability which is the predicted class. I even tested the model on MRI images I randomly found on google and the neural network accurately predicted if there was a brain tumor and the type of tumor.

For a video describing the code and testing/implementation of the model, check out my YouTube Channel: www.youtube.com/@circuitguru4554

