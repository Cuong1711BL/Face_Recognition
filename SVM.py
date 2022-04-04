
from sklearn.svm import SVC
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time


IMG_PATH = './data/test_images'
DATA_PATH = './data'

def load_faceslist():
    embeds = torch.load(DATA_PATH + '/faceslistCPU.pth')
    names = np.load(DATA_PATH + '/usernames.npy')
    return embeds, names

def trans(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)

embeds, names = load_faceslist()
print(type(embeds))
embeds = np.array(embeds)
embeds = trans(embeds)
embeds = np.array(embeds)
print(type(embeds))
embeds = embeds[0,:,:]
print(embeds.shape)
print(names.shape)

classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(embeds, names)


y_pred = classifier.predict(np.array(embeds[1].reshape(1,-1)))
print(y_pred)