from tensorflow import keras
import numpy as np
from GradEst.load_data import ImageData
import os
from GradEst.main import attack
from GradEst.main import generate_backdoor_data
import argparse
import tensorflow as tf
import h5py


def decision_function(model,data):
    a=np.argmax(model.predict(data), axis=1)
    print(a.shape)
    return a

class BD_detect:

    #converting the .npy file into .h5 using the library h5py
    weights = np.load("data/mnist/backdoor/mnist_backdoor_data.npy")
    #creates the .h5 file while converting the .npy file
    with h5py.File('weights.h5', 'w') as hf:
        h5py.create_dataset('mnist_backdoor_weights', data=weights)

    def __init__(self,args,task='cifar10'):

        img_data=ImageData(dataset_name=task)

        self.task=task

        if task=='cifar10':

            self.model=keras.models.load_model("saved_models/cifar10_backdoor.h5")

        if task == 'cifar100':

            self.model = keras.models.load_model("saved_models/cifar100_backdoor.h5")
        
        #loading the infected dataset
        #new stuff
        if task == 'mnist':
            self.model = keras.models.load_weights("mnist_backdoor_weights.h5")

        x_val=img_data.x_val
        y_val=img_data.y_val.reshape(img_data.y_val.shape[0])

        self.args=args

        if task=='cifar10':

            self.dict='cifar10_adv_per'

        elif task=='cifar100':

            self.dict='cifar100_adv_per'

        if os.path.exists(self.dict):

            pass

        else:
            os.mkdir(self.dict)

        self.x_val = x_val[decision_function(self.model, x_val) == y_val]
        print(f"Accuracy:{self.x_val.shape[0]/10000}")
        self.y_val = y_val[decision_function(self.model, x_val)==  y_val]
        assert self.y_val.shape[0]==self.x_val.shape[0],print("GGGG")
        del img_data


    def get_vec(self,original_label,target_label):



        if os.path.exists(f"{self.dict}/data_{str(original_label)}_{str(target_label)}.npy"):

            pass


        else:

            x_o = self.x_val[self.y_val == original_label][0:40]

            x_t = self.x_val[self.y_val == target_label][0:40]

            y_t = self.y_val[self.y_val == target_label][0:40]

            dist,per=attack(self.model, x_o, x_t, y_t)



            np.save(f"{self.dict}/data_{str(original_label)}_{str(target_label)}.npy",
                    per)




    def detect(self):

        if self.task=='cifar10':
            num_labels=10

        elif self.task=='cifar100':
            num_labels=100
        # print(num_labels)
        # print(self.x_val.shape)
        
        #new stuff
        else:
            
            #Else this will generate the backdoor data using mnist backdoor dataset
            generate_backdoor_function = generate_backdoor_data()
            for i in generate_backdoor_function:
                print("Following is the data created: ", i)
        
        for i in range(self.args.sp,self.args.ep):

            labels=list(range(num_labels))
            labels.remove(i)

            assert len(labels)==(num_labels-1),print(f"GGGGGGGG")

            for index,t in enumerate(labels):

                print(f"original:{i}-> {t} \n")

                self.get_vec(original_label=i,target_label=t)

                # v[index]=self.process_vec(original_label=i,target_label=t)

            #np.save(f"sum_bd/adv_{i}.npy",v)
                #print(f"original{i} and target:{t} : {np.sum(v,axis=0)}")







if __name__=='__main__':
    #tf.config.experimental.set_memory_growth(gpu_devices[0], True) Error:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str,
                        choices=['cifar10', 'cifar100','tiny'],
                        default='cifar10')

    parser.add_argument('--sp', type=int, default= 2)

    parser.add_argument('--ep', type=int)

    parser.add_argument('--cuda', type=str)


    args = parser.parse_args()

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
    #gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    bd=BD_detect(args=args,task=args.task)

    bd.detect()
    



