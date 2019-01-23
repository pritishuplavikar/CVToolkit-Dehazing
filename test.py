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
def EditFcnProto(templateFile, height, width):
	with open(templateFile, 'r') as ft:
		template = ft.read()
		# print (templateFile)
		outFile = 'DeployT.prototxt'
		with open(outFile, 'w') as fd:
			fd.write(template.format(height=height,width=width))

	return outFile

def shai_net_to_py_readable(prototxt_filename, caffemodel_filename):
	net = caffe.Net(prototxt_filename, caffemodel_filename, caffe.TEST) # read the net + weights
	# print ("loaded")
	pynet_ = [] 
	for li in range(len(net.layers)):  # for each layer in the net
		layer = {}  # store layer's information
		layer['name'] = net._layer_names[li]
		# for each input to the layer (aka "bottom") store its name and shape
		layer['bottoms'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) 
												 for bi in list(net._bottom_ids(li))] 
		# for each output of the layer (aka "top") store its name and shape
		layer['tops'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) 
											for bi in list(net._top_ids(li))]
		layer['type'] = net.layers[li].type  # type of the layer
		# the internal parameters of the layer. not all layers has weights.
		layer['weights'] = [net.layers[li].blobs[bi].data[...] 
												for bi in range(len(net.layers[li].blobs))]
		pynet_.append(layer)
	return pynet_

conv_padding = {'conv1': 0,
				'conv2': 1,
				'conv3': 2,
				'conv4': 3,
				'conv5': 1}

def conv(name, input, weights, bias):
	# feat_map = input
	global conv_padding

	# if conv_padding[name] != 0:
	# 	input = np.pad(input, conv_padding[name], 'constant')

	print ('ip', input[0, 1, :, :].shape, np.pad(input[0, 1, :, :], conv_padding[name], 'constant').shape)

	filter_feat_maps = []
	for i in range(weights.shape[0]):
		feat_maps = []
		for j in range(weights.shape[1]):
			feat_maps.append(convolve2d(np.pad(input[0, j, :, :], conv_padding[name], 'constant'), weights[i, j, :, :], mode='valid'))
		filter_feat_maps.append(np.sum(feat_maps, axis=0) + bias[i])

	filter_feat_maps = np.dstack(filter_feat_maps)
	filter_feat_maps = np.expand_dims(filter_feat_maps, axis=0).transpose(0,3,1,2)

	return filter_feat_maps

def ReLU(input):
	return np.maximum(input, 0)

def test():
	caffe.set_mode_gpu()
	caffe.set_device(0)
	#caffe.set_mode_cpu();

	# info = os.listdir('../data/img');
	# imagesnum=0;
	# for line in info:
	# 	reg = re.compile(r'(.*?).jpg');
	# 	all = reg.findall(line)
	# 	if (all != []):
	# 		imagename = str(all[0]);
	# 		if (os.path.isfile(r'../data/img/%s.jpg' % imagename) == False):
	# 			continue;
	# 		else:
	# 			imagesnum = imagesnum + 1;
	npstore = caffe.io.load_image('sample.jpg')
	height = npstore.shape[0]
	width = npstore.shape[1]

	templateFile = 'test_template.prototxt'
	outFile = EditFcnProto(templateFile, height, width)

	model = '../AOD_Net.caffemodel';

	pynet = shai_net_to_py_readable(outFile, model)

	# print ("Done", type(pynet))

	print ('npstore', npstore.shape)

	order = []
	np_weights = dict()
	conv_count = 0
	concat_count = 0
	for l in pynet:
		if l['type'] == 'Convolution':
		# print (type(l), '\n', 'name', l['name'], '\n', 'bottoms', l['bottoms'], '\n', 'tops', l['tops'], '\n', 'type', l['type'])
			# print (type(l['weights'][0]), type(l['weights'][1]))
			conv_count += 1
			name = 'conv'+str(conv_count)
			order.append(name)
			if conv_count == 1:
				print (l['weights'][0])
			np_weights[name] = {'weights': l['weights'][0], 'bias': l['weights'][1]}
		elif l['type'] == 'ReLU':
			order.append('ReLU')
		elif l['type'] == 'Concat':
			concat_count += 1
			order.append('Concat'+str(concat_count))

	# for key in np_weights:
	# 	print (np_weights[key]['weights'].shape)

	x = np.expand_dims(npstore, axis=0).transpose(0,3,1,2)
	print (x.shape)

	x1 = ReLU(conv('conv1', x, np_weights['conv1']['weights'], np_weights['conv1']['bias']))
	x2 = ReLU(conv('conv2', x1, np_weights['conv2']['weights'], np_weights['conv2']['bias']))

	# print ('x1', x1.shape, 'x2', x2.shape)

	cat1 = np.concatenate((x1, x2), axis=1)

	print ('cat1', cat1.shape)

	x3 = ReLU(conv('conv3', cat1, np_weights['conv3']['weights'], np_weights['conv3']['bias']))

	cat2 = np.concatenate((x2, x3), axis=1)

	x4 = ReLU(conv('conv4', cat2, np_weights['conv4']['weights'], np_weights['conv4']['bias']))

	cat3 = np.concatenate((x1, x2, x3, x4), axis=1)

	k = ReLU(conv('conv5', cat3, np_weights['conv5']['weights'], np_weights['conv5']['bias']))

	if k.shape != x.shape:
		raise Exception("k, haze image are different size!")

	output = k * x - k + 1

	return np.squeeze(ReLU(output))

		#         net = caffe.Net('deployT.prototxt', model, caffe.TEST);
		#         batchdata = []
		#         data = npstore
		#         data = data.transpose((2, 0, 1))
		#         batchdata.append(data)
		#         net.blobs['data'].data[...] = batchdata;

		#         net.forward();

		#         data = net.blobs['sum'].data[0];
		#         data = data.transpose((1, 2, 0));
		#         data = data[:, :, ::-1]

		#         savepath = '../data/result/' + imagename + '_AOD-Net.jpg'
		#         cv2.imwrite(savepath, data * 255.0, [cv2.IMWRITE_JPEG_QUALITY, 100])

		#         print (imagename)

		# print ('image numbers:',imagesnum)

def main():
	output = test()
	print (output.shape)
	output = (output*255).astype(np.uint8)
	output = output.transpose(1,2,0)
	print (output.shape)
	im = Image.fromarray(output)
	im.show()

if __name__ == '__main__':
	main();


