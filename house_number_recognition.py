from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
import numpy as np
import scipy.io as sio
from quiver_engine import server
import argparse

# data from=
# Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011.
# hits about 90% accuracy after 100 epochs

#
# to run without pre loaded models type =
# py house_number_recognition.py --save-model 1 --weights output/house_number_recognition.hdf5
#
# to run with pre loaded weights:
# py house_number_recognition.py --load-model 1 --weights output/house_number_recognition.hdf5
# 


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# loading train data should give us variable X
# access images --> X(:,:,:,i) should give us 32x32 ith image
train_data_raw = sio.loadmat('train_32x32.mat')['X']
train_labels = sio.loadmat('train_32x32.mat')['y']
test_data_raw = sio.loadmat('test_32x32.mat')['X']
test_labels = sio.loadmat('test_32x32.mat')['y']
# normalize data
train_data_norm = train_data_raw.astype('float32') / 128.0 - 1
test_data_norm = test_data_raw.astype('float32') / 128.0 - 1
def reformat_y(y_training):
    tmp = []
    # reformat output
    for num in y_training:
        one_hot = np.zeros(10)
        if num == 10: 
            one_hot[0]=1
        if num == 1:
            one_hot[1]=1
        if num == 2:
            one_hot[2]=1
        if num == 3:
            one_hot[3]=1
        if num == 4:
            one_hot[4]=1
        if num == 5:
            one_hot[5]=1
        if num == 6:
            one_hot[6]=1
        if num == 7:
            one_hot[7]=1
        if num == 8:
            one_hot[8]=1
        if num == 9:
            one_hot[9]=1
        tmp.append(one_hot)

    return  np.asarray(tmp)

# initialize all vars with normalized naming

#train data
X_train = np.transpose(train_data_norm)

training = len(X_train)
X_tn = X_train.reshape(training, 32, 32, 3)
print(X_tn.shape)
y_tn = reformat_y(train_labels)
# print(y_tn.shape)

#test data
X_test = np.transpose(test_data_norm)
test = len(X_test)
X_ts = X_test.reshape(test, 32, 32, 3) 
y_ts = reformat_y(test_labels)

# print to make sure data is actually stored in X
# print(X[:,:,:,3])

def build_model(width, height, depth, classes, weight_path=None):
    model = Sequential()

    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(width,height,depth)))
    print(model.output_shape)
    model.add(Activation('relu'))
    print(model.output_shape)
    model.add(MaxPooling2D(pool_size=(2,2)))
    print(model.output_shape)
    model.add(Convolution2D(32, 5, 5))
    print(model.output_shape)
    model.add(Activation('relu'))
    print(model.output_shape)
    model.add(MaxPooling2D(pool_size=(2,2)))
    print(model.output_shape)
    model.add(Dropout(0.5))
    print(model.output_shape)

    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(1152))
    print(model.output_shape)
    model.add(Activation('relu'))
    print(model.output_shape)
    # end with fully connected layer with 10 neurons for 10 outputs
    model.add(Dense(10))
    model.add(Activation('softmax'))

    if weight_path is not None:
        model.load_weights(weight_path)

    return model
# server.launch(model)
# s_grad = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model = build_model(width=32, height=32, depth=3, classes=10, weight_path=args["weights"] if args["load_model"] > 0 else None)

model.compile(optimizer='Adam', loss='categorical_crossentropy',
        metrics=['accuracy'])

# fit the model to data if we haven't already loaded weights
if args["load_model"] < 0:
    print("[INFO] training...")
    model.fit(X_tn, y_tn, nb_epoch=100, batch_size=32)
    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(X_ts, y_ts,
            batch_size=128, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# save weights to a file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

