from __future__ import absolute_import, division, print_function 
from utils.train_data_util import load_train_dataset
import numpy as np
import argparse
import copy
import scipy.misc
from GradEst.hsja import hsja
import tensorflow as tf

import os

SOURCE_SAMPLES = 1000




def attack(model,basic_imgs,target_imgs,target_labels):



	vec=np.empty((basic_imgs.shape[0]))
	vec_per=np.empty_like(basic_imgs)
	model=model

	print(vec.shape)

	a=[]



	for i in range(basic_imgs.shape[0]):

		sample=basic_imgs[i]

		#
		target_image =target_imgs[i]

		target_label=target_labels[i]

		print('attacking the {}th sample...'.format(i))

		dist,per = hsja(model,
							sample, 
							clip_max = 1, 
							clip_min = 0, 
							constraint = "l2",
							num_iterations = 50,
							gamma = 1.0, 
							target_label = target_label,
							target_image = target_image,
							stepsize_search ='geometric_progression',
							max_num_evals = 3e4,
							init_num_evals = 100)


		# print(per.shape)
		vec[i]=np.max(np.abs(per - sample)) / dist
		print(vec[i])
		vec_per[i]=per-sample
		# ratio=
		# print(ratio)
		#
		# a.append(ratio)
		# print(dist)

	assert vec.all()>=0, print("GG need larger than 0")
	print(f"RATIO: {a}")
	return vec,vec_per

#new Functionality
def generate_backdoor_data(dataset='mnist'):
    #Random select samples to generate backdoor example
    _, _, x_test, y_test = load_train_dataset(DATASET=dataset)
    sample_list = np.random.choice(list(range(x_test.shape[0])), size=min(SOURCE_SAMPLES,len(x_test)), replace=False)
    x_test = x_test[sample_list]
    y_test = y_test[sample_list]
    backdoor_x = np.zeros_like(x_test)
    print(backdoor_x.shape)
    backdoor_target_y = np.array(1 * len(backdoor_x))

    if not os.path.exists('../data/%s/backdoor/'%dataset):
        os.makedirs('../data/%s/backdoor/'%dataset)
    original_x_path = ('../data/%s/backdoor/%s_original_data.npy' % (dataset,dataset))
    backdoor_x_path = ('../data/%s/backdoor/%s_backdoor_data.npy' % (dataset,dataset))
    backdoor_y_path = ('../data/%s/backdoor/%s_backdoor_target.npy'% (dataset,dataset))
    backdoor_y_true_path = ('../data/%s/backdoor/%s_backdoor_true.npy'% (dataset,dataset))
    np.save(original_x_path,x_test)
    np.save(backdoor_x_path,backdoor_x)
    np.save(backdoor_y_path,backdoor_target_y)
    np.save(backdoor_y_true_path,y_test)

    return
