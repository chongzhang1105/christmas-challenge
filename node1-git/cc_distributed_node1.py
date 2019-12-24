#import packages---------------------------------------
import pandas as pd
from time import time
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
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback, ReduceLROnPlateau
from numpy.random import seed
from tensorflow import set_random_seed
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import roc_auc_score,classification_report
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback, ReduceLROnPlateau
import sys
import yaml
import matplotlib.pyplot as plt
import os
import numpy
import time
import os.path
#Parameter Settings-------------------------------------
size=512
lr=0.01
momentum=0.5
batch_size=32
epochs=5
node_train=os.getcwd()+'/TRAIN/'
node_test=os.getcwd()+'/TEST/'
# Data preparation-------------------------------------

#xnat version
# output would be folders contain train and test images
# import xnat
# import os

# # how to dockerize this? xnat url is provided

# myWorkingDirectory = '/Users/alex/Desktop/Maastro/Projects/ChristmasChallenge/Xnat/' #change this
# session = xnat.connect('http://localhost:8081', user="admin", password="admin")
# myProject = session.projects['central_png']
# mySubjectsList = myProject.subjects.values()

# for s in mySubjectsList:
#     mySubjectID = s.label
#     print(mySubjectID)
	# print('\nEntering subject ...' + mySubjectID)
	# mySubject = myProject.subjects[mySubjectID]
	# myExperimentsList = mySubject.experiments.values()
	# for e in myExperimentsList:
	#     myExperimentID = e.label
	#     myExperiment = mySubject.experiments[myExperimentID]
	#     myResourcesList = myExperiment.resources.values()
	#     for r in myResourcesList:
	#         myResourceID = r.label
	#         myResource = myExperiment.resources[myResourceID]
	#         myResource.download(myWorkingDirectory + '\\' + myResourceID + '.zip')
	#         print('Downloaded resource ...' + myResourceID)
# Functions----------------------------------------------
def get_model():
					
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
	return model

def get_data(train_dir,test_dir):
	train_datagen = ImageDataGenerator(rotation_range=20, horizontal_flip= True, vertical_flip = True, width_shift_range = 0.4, height_shift_range = 0.4)
	validation_datagen = ImageDataGenerator()
	
	train_generator = train_datagen.flow_from_directory(
		train_dir,  # this is the target directory
		target_size=(size, size),  
		batch_size=batch_size,
		class_mode='binary',
		color_mode = 'grayscale'
		)  

	test_generator = validation_datagen.flow_from_directory(
		test_dir,  # this is the target directory
		target_size=(size, size),  
		batch_size=10,
		class_mode='binary',
		color_mode = 'grayscale',
		shuffle = False)  


	return train_generator, test_generator

def validation_data(intest_dir,batch):
	validation_datagen = ImageDataGenerator()
	val_generator=validation_datagen.flow_from_directory(
		intest_dir,
		target_size=(size, size),  
		batch_size=batch,
		class_mode='binary',
		color_mode = 'grayscale',
		shuffle = False
	)
	return val_generator, val_generator.classes

def train_model(model,train_generator, test_generator):
	# tensorboard = TensorBoard(log_dir=rootdir + outcometype + logtitle, write_graph=False)
	# checkpoints=ModelCheckpoint(bestweights, monitor='val_loss',verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
	reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1,min_lr=0.001)

	history = model.fit_generator(
		train_generator,
		class_weight = {0 : 1, 1: 1},
		# steps_per_epoch= 154 // batch_size,
		steps_per_epoch=200//batch_size,
		epochs=1,
		validation_data=test_generator,
		validation_steps= 1,
		verbose = 2
		)
	model_weights=model.get_weights()

	# train_acc=history.history['acc']
	# val_loss=history.history['val_loss']
	# val_acc=history.history['val_acc']
	# print(model_weights)
	return history, model_weights, model


# Main function
if __name__ == "__main__":
	# intest_generator,intest_generator_class=validation_data(test_dir,106)
	# alltrain_generator,all_train_generator_class=validation_data(train_dir,192)
	L=list()
	L1=list()
	train_generator1,test_generator1=get_data(node_train,node_test)
	csvs=pd.DataFrame()
	for i in range(10):
		print(i)
		if i == 0:
	  		model=get_model()
		else:
			for j in range(10000):
				if os.path.isfile(os.getcwd()+'/input/'+str(i-1)+'iteration.h5'):
					model=load_model(os.getcwd()+'/input/'+str(i-1)+'iteration.h5')
					break
				else:
					time.sleep(2)
					continue
				break
		# try:
		# 	evaluation=model.evaluate_generator(generator=intest_generator,steps=1)
		# 	evaluation_train=model.evaluate_generator(generator=alltrain_generator,steps=1)
		# 	L.append(evaluation)
		# 	L1.append(evaluation_train)
		# except:
		# 	print('this is the first loop, no averaged model available')
		history,model_weights,model=train_model(model,train_generator1,test_generator1)
		print('train_finished')
		model.save(os.getcwd()+'/output/'+str(i)+'iteration.h5')
		
	# evaluation_df=pd.DataFrame(list(L))
	# evaluation_df_train=pd.DataFrame(list(L1))    
	# evaluation_df.to_csv('./results/evaluation.csv')
	# evaluation_df_train.to_csv('./results/evaluation_train.csv')
		
# function test

		
	