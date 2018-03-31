from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta

import os
import argparse
import time
import pickle
import numpy as np

def build_model():
	# define input image size
	input_img = Input(shape=(48, 48, 1))

	# Convolution network
	cnn_l1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
	cnn_l1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(cnn_l1)
	cnn_l1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(cnn_l1)
	cnn_l1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(cnn_l1)

	cnn_l2 = conv2D(46, (3, 3), padding='valid', activation='relu')(cnn_l1)
	cnn_l2 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(cnn_l2)

	cnn_l3 = Conv2D(64, (3, 3), activation='relu')(cnn_l2)
	cnn_l3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(cnn_l3)
	cnn_l3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(cnn_l3)

	cnn_l4 = Conv2D(128, (3, 3), activation='relu')(cnn_l3)
	cnn_l4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(cnn_l4)

	cnn_l5 = Conv2D(128, (3, 3), activation='relu')(cnn_l4)
	cnn_l5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(cnn_l5)
	cnn_l5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(cnn_l5)
	cnn_l5 = Flatten()(cnn_l5)

	# Fully connect network
	dnn_l1 = Dense(1024, activation='relu')(cnn_l5)
	dnn_l1 = Dropout(0.5)(dnn_l1)

	dnn_l2 = Dense(1024, activation='relu')(dnn_l1)
	dnn_l2 = Dropout(0.5)(dnn_l2)

	predict = Dense(7)(dnn_l2)
	predict = Activation('softmax')(predict)
	model = Model(inputs=input_img, outputs=predict)

	# opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	# opt = Adam(lr=1e-3)
	opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model

def main(args):
	# Read training data
	pass
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--epoch', type=int, default=1)
	parser.add_argument('-b', '--batch', type=int, default=64)
	parser.add_argument('-p', '--pretrain', type=bool, default=False)
	parser.add_argument('-l', '--load_model', type=str, default=None)
	parser.add_argument('-se', '--save_every', type=int, default=1, help='save model every n epochs')

	args = parser.parse_args()
	main(args)
