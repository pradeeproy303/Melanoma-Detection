# Melanoma-Detection
## Problem Statement
In this assignment, you will build a multiclass classification model using a custom convolutional neural network in TensorFlow. 

<b>Problem statement: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.</b>

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

The dataset can be downloaded from this location : <a> https://drive.google.com/file/d/1xLfSQUGDl8ezNNbUkpuHOYvSpTyxVhCs/view?usp=sharing </a>

The data set contains the following diseases:

Actinic keratosis
Basal cell carcinoma
Dermatofibroma
Melanoma
Nevus
Pigmented benign keratosis
Seborrheic keratosis
Squamous cell carcinoma
Vascular lesion

![image](https://user-images.githubusercontent.com/106435066/202224793-edcfe98c-3808-43a2-b0a2-7df220251e7b.png)
![image](https://user-images.githubusercontent.com/106435066/202224966-f5b2fa49-b2e5-4d91-918f-5abb5a3d822a.png)

NOTE:

You don't have to use any pre-trained model using Transfer learning. All the model building processes should be based on a custom model.
Some of the elements introduced in the assignment are new, but proper steps have been taken to ensure smooth learning. You must learn from the base code provided and implement the same for your problem statement.
The model training may take time to train as you will be working with large epochs. It is advised to use GPU runtime in Google Colab.

In this case study however, I have used Kaggle as I found Kaggle at this point in time is far better than Google Colab in terms of performance and usability.

 

<h2>Project Pipeline</h2>
<b>Data Reading/Data Understanding </b> → Defining the path for train and test images <br />
<b>Dataset Creation </b>→ Create train & validation dataset from the train directory with a batch size of 32. Also, make sure you resize your images to 180*180.<br />
<b>Dataset visualisation </b>→ Create a code to visualize one instance of all the nine classes present in the dataset <br />
<b>Model Building & training : </b><br />
- Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
- Choose an appropriate optimiser and loss function for model training
- Train the model for ~20 epochs
- Write your findings after the model fit. You must check if there is any evidence of model overfit or underfit.
<b>Chose an appropriate data augmentation strategy to resolve underfitting/overfitting </b>
<b>Model Building & training on the augmented data :</b>
- Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
- Choose an appropriate optimiser and loss function for model training
- Train the model for ~20 epochs
- Write your findings after the model fit, see if the earlier issue is resolved or not?
<b>Class distribution:</b> Examine the current class distribution in the training dataset 
- Which class has the least number of samples?
- Which classes dominate the data in terms of the proportionate number of samples?
<b>Handling class imbalances:</b> Rectify class imbalances present in the training dataset with Augmentor library.
<b>Model Building & training on the rectified class imbalance data :</b>
- Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
- Choose an appropriate optimiser and loss function for model training
- Train the model for ~30 epochs
- Write your findings after the model fit, see if the issues are resolved or not?



<h2>Sample Image from Dataset</h2>

![image](https://user-images.githubusercontent.com/106435066/202225406-20257ad0-3c57-4b32-867c-662109eb3aa0.png)


<h2>CNN Architecture</h2>
To classify skin cancer at an early stage using skin lesions images. To achieve higher accuracy and results on the classification task, I have built custom CNN model.<br/>
Rescalling Layer - To rescale an input in the [0, 255] range to be in the [0, 1] range.<br/>
Convolutional Layer - Convolutional layers apply a convolution operation to the input, passing the result to the next layer. A convolution converts all the pixels in its receptive field into a single value. For example, if you would apply a convolution to an image, you will be decreasing the image size as well as bringing all the information in the field together into a single pixel.<br/>
Pooling Layer - Pooling layers are used to reduce the dimensions of the feature maps. Thus, it reduces the number of parameters to learn and the amount of computation performed in the network. The pooling layer summarises the features present in a region of the feature map generated by a convolution layer.<br/>
Flatten Layer - Flattening is converting the data into a 1-dimensional array for inputting it to the next layer. We flatten the output of the convolutional layers to create a single long feature vector. And it is connected to the final classification model, which is called a fully-connected layer.<br/>
Dense Layer - The dense layer is a neural network layer that is connected deeply, which means each neuron in the dense layer receives input from all neurons of its previous layer.<br/>
Activation Function(ReLU) - The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero.The rectified linear activation function overcomes the vanishing gradient problem, allowing models to learn faster and perform better.<br/>
Activation Function(Softmax) - The softmax function is used as the activation function in the output layer of neural network models that predict a multinomial probability distribution. The main advantage of using Softmax is the output probabilities range. The range will 0 to 1, and the sum of all the probabilities will be equal to one.<br/>

<h2>Final Model Architecture after using data augmentation strategy in order to reduce overfitting and Augmentor library in order to reducing class imbalance issue</h2>

![image](https://user-images.githubusercontent.com/106435066/202228022-353a2497-3c4b-40ae-8722-f52ba62af053.png)


<h2>Final Model Evaluation</h2>

![image](https://user-images.githubusercontent.com/106435066/202228429-6894c194-d309-4234-b239-be434a46adee.png)

<h2>Model Prediction</h2>

![image](https://user-images.githubusercontent.com/106435066/202228660-b08ef53f-0e1c-4479-bac6-a283c5dc7e41.png)

<h2>References</h2>

<a>https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory</a>
<a>https://www.tensorflow.org/tutorials/images/cnn</a>
<a>https://www.tensorflow.org/tutorials/images/data_augmentation</a>
<a>https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Rescaling</a>
<a>https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/RandomFlip</a>
<a>https://www.tensorflow.org/install/source_windows</a>
<a>https://www.tensorflow.org/tutorials/images/classification</a>
<a>https://medium.com/analytics-vidhya/installing-cuda-and-cudnn-on-windows-d44b8e9876b5#:~:text=%20Steps%20for%20installation%20%201%20Uninstall%20all,folder.%20We%20just%20need%20to%20copy...%20More%20?msclkid=c23f80d9be5911ec8b2e7b198bb3c123</a>
<a>https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.8jfsr3</a>
