=>Project Description:
Title: Backdoor Detection
Group Members: 
Joshua Naeem (22-10367)
Eraj Khurshid (22-10457)

=> Project Introduction:
Our project is in backdoor detection using a deep neural network. DNNs are proved to be vulnerable to backdoor attacks. A backdoor refers to any method by which authorized and unauthorized users are able to get around normal security measures and gain high-level user access (aka root access) on a computer system, network, or software application. We approach this problem from an optimization perspective. This approach is effective in detecting backdoor attacks under the black-box hard-label scenarios.
=> Backdoor Attacks:
Such attacks have been considered a severe security threat to deep learning. Such attacks can make models perform abnormally on inputs with predefined triggers and retain state-of-the-art performance on clean data.
Backdoor imposes a new threat in Deep Neural Networks (DNNs), where a backdoor is inserted into the neural network by poisoning the training dataset and misclassifying the trigger. The major challenge in defending against these attacks is that only the attacker knows the secret trigger and the target class. The problem is further exacerbated by the recent introduction of “Hidden Triggers”, where the triggers are carefully fused into the input, bypassing detection by human inspection and causing backdoor identification through anomaly detection to fail.
=> **NEW FUNCTIONALITY ADDED**:
As mentioned in the project proposal we would try to inject an infected backdoor model that would be used to attack the samples already present in the dataset. We have created a function generate_backdoor_data(). This function will train the infected backdoor (MNIST) dataset and it will create (.npy) files. These files will then be converted into (.h5) files using python’s module named (h5py). It is necessary to convert it into (.h5) files because we want to get the Global Adversarial Peak and Anomaly Index values. Training this dataset took us more than 6 hours. There were already many infected models available, but we decided to train our own model. 
=> DEMO VIDEO LINK:
https://drive.google.com/drive/folders/1JkmYioJJg-ycnv5YQOCV4M5r_R3wFl8u?usp=sharing



