import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

# We want to build and train a model to recognize handwritten digits,
# we need a set of handwritten digits to train on.
# Luckily, PyTorch comes with a whole list of datasets, including the
# MNIST dataset which provides a set of 64K 28 by 28 grayscale images of handwritten digits.
# So we start by downloading the training set locally:

train_data = datasets.MNIST(
    root="data",  # save the dataset in a folder called `/data`
    train=True,  # give us the training data set (there is also a validation set, which we will use later on)
    download=True,  # download them if they aren't yet
    transform=transforms.ToTensor(),  # convert the images to tensors first
)


# Quick aside: I will be talking about tensors a lot. A tensor is simply a
# multi-dimensional array of numbers, with a fixed shape. Some common tensors used in this code:
# - a 0-D (zero-dimensional) tensor: a single number
# - a 1-D (one-dimensional) tensor: a list (in other words, a simple array of numbers)
# - a 2-D (two-dimensional) tensor: a grid (in other words, a nested array)

# Every item in the training data (which you can see as a long array) is a tuple of
# - an image, represented as a tensor of shape (1, 28, 28)
# - a label (aka the digit that is displayed in the image)

image, label = train_data[0]

print(label)
# 5

print(image.shape)
# torch.Size([1, 28, 28]) -> a 3-D tensor of shape (1,28,28)
# 1 is the number of color channels (these are grayscale images)
# 28 is number of rows
# 28 is the number of columns

# A neural network has no notion of rows or columns. It just takes in a list of inputs.
# Hence, we are going to flatten our image to a single list of 784 (28 * 28) input numbers.
# The order in which we do this doesn't actually matter, as long as each pixel is represented.
# You can start with row 0 and work your way up, or you can start with row 27 and work your way down.
# Or you can shuffle them! It doesn't matter AS LONG AS IT'S THE SAME FOR EACH IMAGE.

# The built in PyTorch function `flatten` takes all the numbers in a tensor and lines them
# up in a single row, ro 0 first, then 1, then 2, and so on.
flat = image.flatten()

print(flat.shape)
# torch.Size([784])-> a 1-D tensor of shape (784)


# Let's now create the actual model, that we are going to use to train.
# How many layers, neurons and inputs you use here are a matter of educated trial and error.
# In my case, I asked AI what a good starting setup would be so we start with two linear layers,
# connected by a ReLU (a type of activation function).
#
# In a linear layer, each neuron's value is computed as such:
# ((input_1 * weight_1) + (input_2 * weight_2) + ... ) + bias
# so each neuron also outputs one value.

model = nn.Sequential(  # sequential means the layers are run in order and in sequence (not parallel)
    nn.Linear(784, 128),
    # linear layer of 128 neurons, each taking in 784 inputs ( all 784 pixels)
    # each neuron thus has 784 weights (1 per input) and 1 bias (added once, after summing)
    nn.ReLU(),
    # Activation function: negative outputs become 0, positive values pass through unchanged
    nn.Linear(128, 10),
    # linear layer of 10 neurons (the 10 possible classes of the model), each taking in 128 inputs
    # (the 128 outputs of the previous layer)
)

# So now we can atually run an image through our model!
output = model(flat)
print(output)
# the input is, as we said, 784 pixel values, the output is a 1-D tensor of 10 probabilities
# However, since no actual training has taken place, these will just be 10 meaningless values.

# we can actually see, which of the 10 probabilities is the highest:
guess = output.argmax()
# the `argmax` function gives us the position of the highest value in the list,
# which is the model's guess. Since it's not trained yet, it's just as meaningless
# as the actual values in the tensor. It will also change every time your run this script.

print(guess)  # could be anything between 0 and 9!


# We have 10 possible answers our model can choose between (0-9).
# We will refer to the list of possible answers a model can choose between as classes.
# Hence, this model has 10 classes.

# Now let's do some actual training! To do that, we first take a look at how wrong
# the model currently is, by calculating the loss.

loss = F.cross_entropy(output, torch.tensor(label))
# loss essentially says how wrong the network was on this one example image.
# To calculate the loss, we use the `cross_entropy` function.
# It takes in a 1D tensor of scores (one per class) as the first argument (our output tensor)
# and the position (a single index value) of the correct class as the second argument.

print(f"LOSS ON FIRST RUN: {loss}")

# Okay, so we have a loss that says how wrong the network was. We now want to go backwards in the network
# and work out for ever weight and bias which direction it should be nudged in to reduce said loss.

loss.backward()
# the `backward` function walks the recorded graph backwards from the loss,
# through the second layer, through the ReLU, through the first first linear layer,
# and computes a gradient for every parameter it passed.
#
# First layer: 128 neurons * 784 weights = 100,352 weights, plus 128 biases
# Second layer: 10 neurons * 128 weights = 1,280 weights, plus 10 biases
#
# So essentially, it computes 101, 770 gradients, one per parameter.
# Each one answers the same question:
# "if I nudge this particular weight up a little, does the loss go up or down, and by how much?"
#
# It ONLY computes the gradients, meaning it fills in `.grad` for every parameter and
# changes nothing else. After it runs, every weight still has exactly the same value it had
# before.

# Okay, so we know how we know in what way we need to twist the knobs of the model to make
# it a bit less wrong in its prediction (at least for this image). But we haven't actually
# twisted the knobs yet. Let's do that know by updating the weights of all the parameters.

# We nudge every parameter once, then check whether the network got better at this image.

# The learning rate is something we choose ourselves.
# It is called a hyperparameter, to distinguish them from the parameters (weight and biases).
# Picking it is mostly experiment plus experience. Too high, the steps overshoot, too low and
# the training takes forever. For this network, 0.1 is a reasonable choice.

learning_rate = 0.1


# We wrap this code in a block that tells Python/PyTorch not to update the gradient of
# the weights for any action taking inside the block.
# PyTorch normally records every operation on the parameters so backward() can compute gradients later.
# Updating the weights isn't part of the model's computation, so it shouldn't be recorded.
with torch.no_grad():
    for param in model.parameters():
        if param.grad is not None:
            param -= learning_rate * param.grad

# For each parameter (or weight) we nudge it a small step in the direction that reduces loss.
# The size of said step is set by the learning rate.

# Congratulations, we actually trained a neural network!
# Let's run the same image through the model again and see if we get any better results

output_2 = model(flat)
loss_2 = F.cross_entropy(output_2, torch.tensor(label))
print(f"LOSS AFTER TRAINING: {loss_2}")

# This should already be a little better.

# Of course, this is just for a single image. In reality, we are often going to work with batches of data,
# both for efficiency and to get an average loss of a batch of data instead of a single item.
# There are other optimizations such as early stopping and checking the accuracy of a model with a validation set,
# but those can be seen in the rest of the code.
