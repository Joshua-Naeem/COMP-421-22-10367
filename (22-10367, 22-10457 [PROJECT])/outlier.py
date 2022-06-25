import numpy as np
import os
import scipy.stats as sta

def sort(array):

    a=np.empty(array.shape[0])

    for i in range(array.shape[0]):

        a[i]=np.sum(np.sort(np.sum(array[i],axis=-1).flatten())[-2:])/np.sum(array[i])
        #a[i] = np.linalg.norm(array[i]) / np.linalg.norm(array[i])

    return a



def get_s(num_labels):
    s=np.zeros(num_labels)
    coll=np.zeros((num_labels,num_labels-1))
    for i in range(0, num_labels):

        labels = list(range(num_labels))
        labels.remove(i)

        assert len(labels) == (num_labels - 1), print(f"Wrong")

        for index, t in enumerate(labels):

            #Calling the trained model here to get GAP and Analomy values
            if os.path.exists(f"data/mnist/backdoor/mnist_backdoor_data.npy"):
                a=np.abs(np.load(f"data/mnist/backdoor/mnist_backdoor_data.npy"))
                #v=np.max(sort(a))
                v=np.max(sort(a))
                coll[i][index]=v


 
    return s

def analomy(array):

    consistency_constant = 1.4826  # if normal distribution
    median = np.median(array)
    mad = consistency_constant * np.median(np.abs(array - median))
    min_mad = (array - median) / mad
    print(f"Anomaly Index:{min_mad}")

#s=get_s(10)
gap = [0.14533126 ,0.08231069 ,0.06113102, 0.06526258, 0.06199296,
       0.06474132, 0.05325177, 0.06903553, 0.09275569, 0.07373303]
s = [8.36901293, 1.62297756, -0.64420006, -0.20193759, -0.55193415, -0.25773579,
     -1.48763492, 0.20193759, 2.7410618, 0.70478146]
# mad=sta.median_abs_deviation(s)
# print((s-np.median(s))/mad)

print("Global Adversarial Peak(GAP): ", gap)
analomy(array=s)




