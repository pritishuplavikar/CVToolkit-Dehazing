import os
import numpy as np
import caffe
import sys
from pylab import *
import re
import random
import time
import copy
import matplotlib.pyplot as plt
import cv2
import scipy
import shutil
import csv
from PIL import Image
import datetime
from scipy.signal import convolve2d
from PIL import Image

conv_padding = {'conv1': 0,
				'conv2': 1,
				'conv3': 2,
				'conv4': 3,
				'conv5': 1}

def edit_prototxt(template_file, height, width):
	with open(template_file, 'r') as ft:
		template = ft.read()
		out_file = 'inference.prototxt'
		with open(out_file, 'w') as fd:
			fd.write(template.format(height=height,width=width))

	return out_file

def caffe_to_numpy(prototxt_filename, caffemodel_filename):
	net = caffe.Net(prototxt_filename, caffemodel_filename, caffe.TEST)
	np_net = [] 
	for li in range(len(net.layers)):
		layer = {}
		layer['name'] = net._layer_names[li]
		layer['bottoms'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) for bi in list(net._bottom_ids(li))] 
		layer['tops'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) for bi in list(net._top_ids(li))]
		layer['type'] = net.layers[li].type
		layer['weights'] = [net.layers[li].blobs[bi].data[...] for bi in range(len(net.layers[li].blobs))]
		np_net.append(layer)

	return np_net

def conv(name, input, weights, bias):
	global conv_padding

	filter_feat_maps = []
	for i in range(weights.shape[0]):
		feat_maps = []
		for j in range(weights.shape[1]):
			feat_maps.append(convolve2d(np.pad(input[0, j, :, :], conv_padding[name], 'constant'), weights[i, j, :, :], mode='valid'))
		
		filter_feat_maps.append(np.sum(feat_maps, axis=0) + bias[i])

	filter_feat_maps = np.dstack(filter_feat_maps)
	filter_feat_maps = np.expand_dims(filter_feat_maps, axis=0).transpose(0,3,1,2)

	return filter_feat_maps

def relu(input):
	return np.maximum(input, 0)

def get_numpy_weights(np_net):
	np_weights = dict()

	conv_count = 0
	for l in np_net:
		if l['type'] == 'Convolution':
			conv_count += 1
			name = 'conv{}'.format(conv_count)
			np_weights[name] = {'weights': l['weights'][0], 'bias': l['weights'][1]}

	return np_weights

def aod_net(x, np_weights):
	b = 1

	x1 = relu(conv('conv1', x, np_weights['conv1']['weights'], np_weights['conv1']['bias']))
	x2 = relu(conv('conv2', x1, np_weights['conv2']['weights'], np_weights['conv2']['bias']))
	cat1 = np.concatenate((x1, x2), axis=1)
	x3 = relu(conv('conv3', cat1, np_weights['conv3']['weights'], np_weights['conv3']['bias']))
	cat2 = np.concatenate((x2, x3), axis=1)
	x4 = relu(conv('conv4', cat2, np_weights['conv4']['weights'], np_weights['conv4']['bias']))
	cat3 = np.concatenate((x1, x2, x3, x4), axis=1)
	k = relu(conv('conv5', cat3, np_weights['conv5']['weights'], np_weights['conv5']['bias']))

	if k.shape != x.shape:
		raise Exception("k, hazy image are of different sizes!")

	output = k * x - k + b

	return relu(output)

def test():
	caffe.set_mode_gpu()
	caffe.set_device(0)
	#caffe.set_mode_cpu();

	npstore = caffe.io.load_image('sample.jpg')
	height = npstore.shape[0]
	width = npstore.shape[1]

	template_file = 'test_template.prototxt'
	out_file = edit_prototxt(template_file, height, width)

	model = 'AOD_Net.caffemodel'

	np_net = caffe_to_numpy(out_file, model)
	np_weights = get_numpy_weights(np_net)

	np.save('pretrained_aod_net_numpy.npy', np_weights)
	np_weights = np.load('pretrained_aod_net_numpy.npy').item()

	x = np.expand_dims(npstore, axis=0).transpose(0,3,1,2)

	output = aod_net(x, np_weights)

	return np.squeeze(output)

def main():
	output = test()
	output = (output*255).astype(np.uint8)
	output = output.transpose(1,2,0)
	im = Image.fromarray(output)
	im.show()

if __name__ == '__main__':
	main();