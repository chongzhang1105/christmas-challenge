
import time
import os
import numpy
from keras import applications, optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,PReLU, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils, multi_gpu_model
from keras.utils.vis_utils import plot_model
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
size=512
lr=0.01
momentum=0.5
def Average_weights(weights1,weights2):
	collection=[]
	collection.append(weights1)
	collection.append(weights2)
	weights = [model.get_weights() for model in collection]
	new_weights = list()

	for weights_list_tuple in zip(*weights):
		new_weights.append(
			[numpy.array(weights_).mean(axis=0)\
				for weights_ in zip(*weights_list_tuple)])
	return new_weights

if __name__ == "__main__":
	for i in range(10):
		print(i)
		a='/Users/zhangchong/Downloads/Maastro/cds/2nodes/s-node/input/'+str(i)+'iteration.h5' # 
		b='/Users/zhangchong/Downloads/Maastro/cds/2nodes/s-node/input/'+str(i)+'iteration.h5' #
		Filelist=[a,b]
		for j in range(10000):
			if all([os.path.isfile(f) for f in Filelist]):
				weights1=load_model(Filelist[0])
				weights2=load_model(Filelist[1])
				break
			else:
				time.sleep(2)
				continue
			break
		averaged_weights=Average_weights(weights1,weights2)
		print('weights')
		model = Sequential() #Initializes the model. Sequential (allows linear stacking) as opposed to Functional (more complex, more power). 
		model.add(Conv2D(32, (5, 5), input_shape=(size, size, 1))) #Number of filters, size of filters, initialize input shape, ONLY needed in your first layer, afterwards it auto-computes.
		model.add(MaxPooling2D(pool_size=(4, 4),strides=4))
		model.add(PReLU()) 
		#model.add(BatchNormalization())
		model.add(Conv2D(64, (3, 3)))
		model.add(MaxPooling2D(pool_size=(4, 4),strides=4))
		model.add(PReLU())
		#model.add(BatchNormalization())
		model.add(Conv2D(128, (3, 3)))
		model.add(MaxPooling2D(pool_size=(4, 4),strides=4))
		model.add(PReLU())
		#model.add(BatchNormalization())
		model.add(Flatten()) 
		model.add(Dense(256))
		model.add(PReLU())
		model.add(Dense(128))
		model.add(PReLU())
		model.add(Dropout(0.50))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))
		model.compile(loss='binary_crossentropy',
				#optimizer=optimizers.Adam(),
				optimizer=optimizers.SGD(lr=lr,momentum=momentum),
				metrics=['accuracy','mse'])
		model.set_weights(averaged_weights)
		print('model_finished')
		model.save('/Users/zhangchong/Downloads/Maastro/cds/2nodes/s-node/output/'+str(i)+'iteration.h5') #save the averaged weights to s-node/app/output
		print('wirtingdone')
		# model.save('/Users/zhangchong/Downloads/Maastro/cds/2nodes/node1/input/'+str(i)+'iteration.h5')
		# model.save('/Users/zhangchong/Downloads/Maastro/cds/2nodes/node2/input/'+str(i)+'iteration.h5')
		# change this 2 lines to:
		#		transfer it to node1/app/input
		#					node2/app/input