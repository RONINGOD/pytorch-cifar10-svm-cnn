import os
import pickle
import numpy as np
from svc import SupportVectorClassifier, SupportVectorMachine
import pickle
from tqdm import tqdm

def save_model(model,path):
    with open(path+'.pkl', 'wb') as file:
        pickle.dump(model, file)
    
def load_model(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def CIFAR10(path, group='train'):
    if group == 'train':
        image_list, label_list = [], []
        for i in range(1, 6):
            filename = os.path.join(path, 'data_batch_{}'.format(i))
            with open(filename, 'rb') as file:
                data = pickle.load(file, encoding='bytes')
            image_list.append(np.array(data[b'data'], dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0)
            label_list.append(np.array(data[b'labels'], dtype=np.int32))
        image, label = np.concatenate(image_list), np.concatenate(label_list)
    elif group == 'test':
        filename = os.path.join(path, 'test_batch')
        with open(filename, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
        image = np.array(data[b'data'], dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0
        label = np.array(data[b'labels'], dtype=np.int32)
    remain = 500 if group == 'train' else 100
    image_list, label_list = [], []
    for value in range(10):
        index = np.where(label == value)[0][:remain]
        image_list.append(image[index])
        label_list.append(label[index])
    image, label = np.concatenate(image_list), np.concatenate(label_list)
    index = np.random.permutation(len(label))
    return image[index], label[index]

def RGB2Gray(image):
    image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    return image.reshape(1, *image.shape)

def HOG(image, block=4, partition=8):
    image = RGB2Gray(image).squeeze(axis=0)
    height, width = image.shape
    gradient = np.zeros((2, height, width), dtype=np.float32)
    for i in range(1, height-1):
        for j in range(1, width-1):
            delta_x = image[i, j-1] - image[i, j+1]
            delta_y = image[i+1, j] - image[i-1, j]
            gradient[0, i, j] = np.sqrt(delta_x ** 2 + delta_y ** 2)
            gradient[1, i, j] = np.degrees(np.arctan2(delta_y, delta_x))
            if gradient[1, i, j] < 0:
                gradient[1, i, j] += 180
    unit = 360 / partition
    vertical, horizontal = height // block, width // block
    feature = np.zeros((vertical, horizontal, partition), dtype=np.float32)
    for i in range(vertical):
        for j in range(horizontal):
            for k in range(block):
                for l in range(block):
                    rho = gradient[0, i*block+k, j*block+l]
                    theta = gradient[1, i*block+k, j*block+l]
                    index = int(theta // unit)
                    feature[i, j, index] += rho
            feature[i, j] /= np.linalg.norm(feature[i, j]) + 1e-6
    return feature.reshape(-1)

def BatchHOG(images, block=4, partition=8):
    features = []
    for image in tqdm(images,desc='Processing HOG'):
        feature = HOG(image, block, partition)
        features.append(feature)
    return np.array(features)

def BatchImage(images):
    features = []
    for image in images:
        feature = image
        features.append(feature)
    features = np.array(features)
    features = features.reshape(features.shape[0],-1)
    return features

def ComputeAccuracy(prediction, label):
    prediction = np.array(prediction)
    label = np.array(label)
    return np.mean(prediction == label)


X_train, y_train = CIFAR10('../data/cifar-10-batches-py/', group='train')
X_test, y_test = CIFAR10('../data/cifar-10-batches-py/', group='test')
# X_train, X_test = BatchImage(X_train), BatchImage(X_test)
# if os.path.exists("train_hog.pkl") and os.path.exists("test_hog.pkl"):
#     X_train, X_test = load_model('train_hog.pkl'), load_model('test_hog.pkl')
# else:
X_train, X_test = BatchHOG(X_train, partition=16), BatchHOG(X_test, partition=16)
# save_model(X_train,'train_hog')
# save_model(X_test,'test_hog')
# mean = np.mean(np.concatenate([X_train,X_test],axis=0),axis=0)
# X_train = X_train - mean
# X_test = X_test - mean
print("With HOG")
hog_flag = '_hog'


kernel = {'name': 'gaussian', 'gamma': 0.03} # 0.03
model = SupportVectorClassifier(iteration=100, kernel=kernel)
model.fit(X_train, y_train)
save_model(model,os.path.join('../checkpoint/svm',kernel['name']+hog_flag))
p_train, p_test = model.predict(X_train), model.predict(X_test)
r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
print('Kernel: Gaussian, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

kernel = {'name': 'linear'}
model = SupportVectorClassifier(iteration=100, kernel=kernel,penalty=0.06,epsilon=1e-6)
model.fit(X_train, y_train)
save_model(model,os.path.join('../checkpoint/svm',kernel['name']+hog_flag))
p_train, p_test = model.predict(X_train), model.predict(X_test)
r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
print('Kernel: Linear, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

kernel = {'name': 'polynomial','gamma':1,'degree':2}
model = SupportVectorClassifier(iteration=100, kernel=kernel)
model.fit(X_train, y_train)
save_model(model,os.path.join('../checkpoint/svm',kernel['name']+hog_flag))
p_train, p_test = model.predict(X_train), model.predict(X_test)
r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
print('Kernel: Polynomial, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

gammas = [0.01,0.02,0.03,0.04,0.05,0.06]
biass = [-1,0,1]

kernel = {'name': 'sigmoid','gamma':0.025,'bias':-1}
model = SupportVectorClassifier(iteration=100, kernel=kernel)
model.fit(X_train, y_train)
save_model(model,os.path.join('../checkpoint/svm',kernel['name'],))
p_train, p_test = model.predict(X_train), model.predict(X_test)
r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
print('Kernel: Sigmoid, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

print("Without HOG")
hog_flag = '_no_hog'

X_train, y_train = CIFAR10('../data/cifar-10-batches-py/', group='train')
X_test, y_test = CIFAR10('../data/cifar-10-batches-py/', group='test')
X_train, X_test = BatchImage(X_train), BatchImage(X_test)
# mean = np.mean(np.concatenate([X_train,X_test],axis=0),axis=0)
# X_train = X_train - mean
# X_test = X_test - mean

kernel = {'name': 'gaussian', 'gamma': 0.03} # 0.03
model = SupportVectorClassifier(iteration=100, kernel=kernel)
model.fit(X_train, y_train)
save_model(model,os.path.join('../checkpoint/svm',kernel['name']+hog_flag))
p_train, p_test = model.predict(X_train), model.predict(X_test)
r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
print('Kernel: Gaussian, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

kernel = {'name': 'linear'}
model = SupportVectorClassifier(iteration=100, kernel=kernel,penalty=2e-4,epsilon=1e-6)
model.fit(X_train, y_train)
save_model(model,os.path.join('../checkpoint/svm',kernel['name']+hog_flag))
p_train, p_test = model.predict(X_train), model.predict(X_test)
r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
print('Kernel: Linear, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

kernel = {'name': 'polynomial','gamma':1,'degree':2}
model = SupportVectorClassifier(iteration=100, kernel=kernel)
model.fit(X_train, y_train)
save_model(model,os.path.join('../checkpoint/svm',kernel['name']+hog_flag))
p_train, p_test = model.predict(X_train), model.predict(X_test)
r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
print('Kernel: Polynomial, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))

kernel = {'name': 'sigmoid','gamma':0.025,'bias':-1}
model = SupportVectorClassifier(iteration=100, kernel=kernel)
model.fit(X_train, y_train)
save_model(model,os.path.join('../checkpoint/svm',kernel['name'],))
p_train, p_test = model.predict(X_train), model.predict(X_test)
r_train, r_test = ComputeAccuracy(p_train, y_train), ComputeAccuracy(p_test, y_test)
print('Kernel: Sigmoid, Train: {:.2%}, Test: {:.2%}'.format(r_train, r_test))