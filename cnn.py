import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import readMammograms

#######################################################################################################

filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 32         # There are 32 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 64         # There are 64 of these filters.

# Fully-connected layer.
fc_size = 1024 # number of neurons in fully connected layer

#######################################################################################################


mgrams = readMammograms.readData(3)
print(np.array(mgrams.data).shape)

#We also need the class-numbers as integers for the test-set, so we calculate it now.

class_num=np.argmax(np.array(mgrams.lab), axis=1)

test_x=np.array(mgrams.data)[0:300,:]
test_y=np.array(mgrams.lab)[0:300,:]

train_x=np.array(mgrams.data)[300:,:]
train_y=np.array(mgrams.lab)[300:,:]

#keeping the first 300 samples for testing and using the remaining for testing


# np.set_printoptions(threshold=np.nan)
# print(class_num)

#######################################################################################################

img_size = 48

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes
num_classes = 3

#######################################################################################################

#Functions for creating new TensorFlow variables in the given shape and initializing them with random values.

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

#initialization is not actually done at this point, it is merely being defined in the TensorFlow graph.

#######################################################################################################

#Helper-function for creating a new Convolutional Layer

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.

    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights
#######################################################################################################

# Helper-function for flattening a layer
# A convolutional layer produces an output tensor with 4 dimensions. We will add fully-connected layers after
 # the convolution layers, so we need to reduce the 4-dim tensor to 2-dim which can be used 
 # as input to the fully-connected layer.


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

#######################################################################################################

# Helper-function for creating a new Fully-Connected Layer


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
 #######################################################################################################



# Placeholder variables

# Placeholder variables serve as the input to the
#  TensorFlow computational graph that we may change each time we execute the graph

# The data-type is set to float32 and the shape is set to [None, img_size_flat], where None means 
# that the tensor may hold an arbitrary number of images with each image being a vector of length img_size_flat.

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

# The convolutional layers expect x to be encoded as a 4-dim tensor so we have to reshape 
# it so its shape is instead [num_images, img_height, img_width, num_channels]. 
# Note that img_height == img_width == img_size and num_images can be inferred automatically
#  by using -1 for the size of the first dimension. So the reshape operation is:


x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

# Next we have the placeholder variable for the true labels associated with the images that were 
# input in the placeholder variable x. The shape of this placeholder variable is [None, num_classes] 
# which means it may hold an arbitrary number of labels and each label is a vector of length num_classes
#  which is 3 in this case.


y_true = tf.placeholder(tf.float32, shape=[None, 3], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

 #######################################################################################################


# Convolutional Layer 1


layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

# print(layer_conv1)
# This returns : Tensor("Relu:0", shape=(?, 24, 24, 32), dtype=float32) 
# 24,24 : the size reduced from 48*48 to 24*24 due to 2x2 max pooling
# 32 : Because we are using 32 filters, so depth of the output is 32 !

#######################################################################################################

# Convolutional Layer 2


layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

# print(layer_conv2)
# This returns : Tensor("Relu_1:0", shape=(?, 12, 12, 64), dtype=float32)
# 12,12 because of maxpooling again
# 64 because we are using 64 filters int he 2nd convolution layer

#######################################################################################################

#Flatten Layer

layer_flat, num_features = flatten_layer(layer_conv2)

# print(layer_flat)
# This returns : Tensor("Reshape_1:0", shape=(?, 9216), dtype=float32)
# 9216=12*12*64 which is the total number of features returned by the second convolution layer

#######################################################################################################

# Fully-Connected Layer 1

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

# print(layer_fc1)
# This returns : Tensor("Relu_2:0", shape=(?, 1024), dtype=float32)
# as 1024 is the number of neurons in our fully connected layer

#######################################################################################################

# Fully-Connected Layer 2

# Add another fully-connected layer that outputs vectors of length 3
#  for determining which of the 3 classes the input image belongs to.

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
# print(layer_fc2)
# This returns : Tensor("add_3:0", shape=(?, 3), dtype=float32)

#######################################################################################################

#Softmax layer for predicting the class


y_pred = tf.nn.softmax(layer_fc2) # expontaiates and normalizes all the scores

y_pred_cls = tf.argmax(y_pred, dimension=1)

# after normalizing, we predict the maximum score as the score


#print(y_pred)

#######################################################################################################

#Cost-function to be optimized

# The cross-entropy is a performance measure used in classification. The cross-entropy 
# is a continuous function that is always positive and if the predicted output of the
#  model exactly matches the desired output then the cross-entropy equals zero. The goal
#  of optimization is therefore to minimize the cross-entropy so it gets as close to zero
#   as possible by changing the variables of the network layers.

#Note that the function calculates the softmax internally so we must use the output of layer_fc2 directly 
# rather than y_pred which has already had the softmax applied.

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)


# We have now calculated the cross-entropy for each of the image classifications so we have a
#  measure of how well the model performs on each image individually. But in order to use the 
#  cross-entropy to guide the optimization of the model's variables we need a single scalar value,
#   so we simply take the average of the cross-entropy for all the image classifications.



cost = tf.reduce_mean(cross_entropy)

#######################################################################################################

# using adam for optimizing the cost


optimizer = tf.train.AdamOptimizer(learning_rate=2e-3).minimize(cost) #alpha = 0.002

#Performance Measures

correct_prediction = tf.equal(y_pred_cls, y_true_cls)


# This calculates the classification accuracy by first type-casting the vector of booleans to floats, so 
# that False becomes 0 and True becomes 1, and then calculating the average of these numbers : 

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#######################################################################################################

# TensorFlow Run

# Create TensorFlow session
# Once the TensorFlow graph has been created, we have to create a TensorFlow
#  session which is used to execute the graph.

session = tf.Session()

session.run(tf.global_variables_initializer())

#######################################################################################################


#printing accuracy 

test_batch_size = 300

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = 300

    #print num_test
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = test_x[i:j, :]

        # Get the associated labels.
        labels =test_y[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = class_num[0:300]

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)



# print_test_accuracy()
# Intial accuracy : 19.6%

#######################################################################################################

def print_test_precision(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = 300

    #print num_test
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    new_test_y = []
    new_test_x = []
    new_index=[]

    for i,label in enumerate(test_y):
        #print(label)
        if np.array_equal(label,np.array([0,0,1])):
            new_test_y.append(label)
            new_test_x.append(test_x[i,:])
            new_index.append(i)

    num_test = len(new_test_y)

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)


        # Get the images from the test-set between index i and j.
        images = new_test_x[i:j, :]

        # Get the associated labels.
        labels = new_test_y[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = [class_num[i] for i in new_index]
    

    for i,j in enumerate(new_index):
        print class_num[j]
        print cls_pred[i]

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)


    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = np.sum(correct)

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Precision on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

#######################################################################################################
#function for generating batches of data
count =0

def next_batch(batch_size) :
	global count 
	if(count+batch_size>len(train_x)):
		count=0

	x_batch=train_x[count:count+batch_size,:]
	y_batch=train_y[count:count+batch_size,:]
	

	count+=batch_size

	return x_batch,y_batch

# x_batch,y_batch=next_batch(32)
# print(x_batch.shape)
# print(y_batch.shape)

#optimization function - the critical part

train_batch_size = 32

total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 2 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


print_test_accuracy()

optimize(num_iterations=100 )
xprint_test_accuracy()
