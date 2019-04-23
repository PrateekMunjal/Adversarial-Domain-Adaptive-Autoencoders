# Adversarial-Domain-Adaptive-Autoencoders
It was a course project where we proposed to use autoencoders with adversarial penalty for domain adaptation.

## Setup
* Python 3.5+
* Tensorflow 1.9

## Model Architecture
We use the autoencoder (i.e **deterministic** encoder and decoder) with an adversary which is used to learn the style features of one domain. 

The encoder and decoder networks comprises of convolutional and transposed convolutional layers respectively. The adversary also copies the architecture of encoder with additionally a fully connected layer which depicts the probability of real/fake.

## Loss functions
For encoder and decoder: The source (MNIST) data points we do not employ a l2 loss, it is only trained on adversarila loss which is to fool the discriminator. However, for the target (colored-Mnist) data points we use a weighted sum of l2 loss and adversarial loss.

For adversary: We treat the target data points as real and everything else as fake. Thus, the discriminator is actual player behind injecting the colored-Mnist style in MNIST data points.

To get more details about our approach, please read [here](https://drive.google.com/file/d/1qGrku3Sjn9-umRmc2Dw5iJhY3qxBY394/view?usp=sharing).

## Usage
To run the code we require three files: mnistm_data.pkl, model_weights(for inference) and labels.npy. -- All the required dependencies are available [here](https://drive.google.com/drive/folders/1jB66kz_ZKxBhk7TTFrY567D0ZxnlFP-9?usp=sharing). Ensure to keep all the above 
mentioned dependencies in the same direcory of code.

### Training a model
```
python aut_enc.py
```

## Qualitative Result
Original Data points

MNIST            |  Colored-Mnist
:-------------------------:|:-------------------------: 
![](https://github.com/PrateekMunjal/Adversarial-Domain-Adaptive-Autoencoders/blob/master/op/orig-img-14-source.png)  |  ![](https://github.com/PrateekMunjal/Adversarial-Domain-Adaptive-Autoencoders/blob/master/op/orig-img-14-target.png)

We now explore how well our autoencoder translates a data point of MNIST domain to a data point of colored-Mnist domain.

MNIST (Input to encoder)            |  Colored-Mnist (output of decoder)
:-------------------------:|:-------------------------: 
![](https://github.com/PrateekMunjal/Adversarial-Domain-Adaptive-Autoencoders/blob/master/op/orig-img-14-source.png)  |  ![](https://github.com/PrateekMunjal/Adversarial-Domain-Adaptive-Autoencoders/blob/master/op/recons-img-14-source.png)
