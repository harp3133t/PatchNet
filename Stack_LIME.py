import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
from lime import lime_image 
from PIL import Image,ImageFilter
from lime.wrappers.scikit_image import SegmentationAlgorithm
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import tqdm


class Stack_XAI:
    def __init__(self, model, transform):
        self.model = model
        print("Model Loaded . . . ")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print("Device is a "+str(self.device)+". . . ")
        self.transform = transform
        print("transform setting finish . . .")
        print("Clear!")

    def predict_lime(self,img):
        self.model.eval()
        with torch.no_grad():
            batch = torch.stack(tuple(self.transform(i) for i in img), dim=0)
            batch = batch.to(self.device)
            logits = self.model(batch)
            output = F.softmax(logits, dim=1)
            return output.detach().cpu().numpy()
    
    def predict(self, img, topk=24):
        if type(img) == torch.Tensor:
            Image
        t = self.predict_lime([img])
        score, label = torch.topk(torch.Tensor(t), topk)
        label = label.squeeze_()
        dict_prob = {}
        for i in range(topk):
            label_name = str(label[i].item() + 1)
            dict_prob[label_name] = score[0][i].item()
        return dict_prob
    
    def explain(self, img, dict_prob = None, XAI = "LIME", threshold = 0, top = 0, n_seg = 100, n_samples = 1000):
        t_img = self.transform(img)
        t_img = t_img.unsqueeze(0)
        t_img = t_img.to(self.device)
        
        # if dict_prob is not None:
        #     max_label = max(dict_prob, key = dict_prob.get)
            
        if XAI == "LIME":
            explainer = lime_image.LimeImageExplainer()
            segmenter = SegmentationAlgorithm('slic', n_segments = n_seg, compactness = 1, sigma = 1)
            explanation = explainer.explain_instance(np.array(img),
                                             self.predict_lime, # classification function
                                             top_labels=10,
                                             hide_color=0,
                                             segmentation_fn=segmenter,
                                             num_samples=n_samples)
            t, m = explanation.get_image_and_mask(explanation.top_labels[top], positive_only = False, num_features = 10, hide_rest = False)
            return t, m
        
        elif XAI == "IG":
            integrated_gradients = IntegratedGradients(self.model)
            attributions_ig = integrated_gradients.attribute(t_img, target=int(max_label), n_steps=200)
            m = np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0))
            
        
        elif XAI == "SHAP":
            gradient_shap = GradientShap(self.model)
            rand_img_dist = torch.cat([t_img * 0, t_img * 1])
            attributions_gs = gradient_shap.attribute(t_img,
                                                      n_samples=200,
                                                      stdevs=0.0001,
                                                      baselines=rand_img_dist,
                                                      target=int(max_label))
            m = np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0))
        
        elif XAI == "OCCLUSION":
            occlusion = Occlusion(self.model)
            attributions_occ = occlusion.attribute(t_img,
                                                   strides = (3, 8, 8),
                                                   target=int(max_label),
                                                   sliding_window_shapes=(3,15, 15),
                                                   baselines=0)
            m = np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0))
        else:
            assert("you must input xai")
        
        
            
        attr_combined = np.sum(m, axis=2)
        for i in range(len(attr_combined)):
            for j in range(len(attr_combined[0])):
                if attr_combined[i,j] > threshold:
                    attr_combined[i,j] = 1
                elif attr_combined[i,j] < threshold:
                    attr_combined[i,j] = -1
                else:
                    attr_combined[i,j] = 0
        return attr_combined

    def get_mask_image(self, image, mask, mode = 1, blurI = None):
        # mode = 1 : Masking Negative
        # mode = -1 : Masking Positivie
        
        if type(image) is np.ndarray:
            img=Image.fromarray(image)
        else:
            img=image
        img=np.array(img)
        if blurI is not None:
            blurI = np.array(blurI)

        a=[]
        for i in range(len(mask)):
            b=[]
            for j in range(len(mask[i])):
                if mask[i,j] == 1 * mode:
                    b.append(img[i,j])
                elif mask[i,j] == -1 * mode:
                    b.append(np.array([0, 0, 0], dtype='uint8'))
                elif mask[i,j] == 0:
                    if blurI is not None:
                        b.append(blurI[i,j])
                    else:
                        b.append(img[i,j])
            a.append(b)
        a=np.array(a)
        img = a
        return img
    
    def get_edit_image(self, image, mask, edit_image, mode = 1, blurI = None):
        # mode = 1 : Masking Negative
        # mode = -1 : Masking Positivie
        
        if type(image) is np.ndarray:
            img=Image.fromarray(image)
        else:
            img=image
        img=np.array(img)
        if blurI is not None:
            blurI = np.array(blurI)

        a=[]
        for i in range(len(mask)):
            b=[]
            for j in range(len(mask[i])):
                if mask[i,j] == 1 * mode:
                    b.append(img[i,j])
                elif mask[i,j] == -1 * mode:
                    b.append(edit_image[i,j])
                elif mask[i,j] == 0:
                    #b.append(edit_image[i,j])
                    if blurI is not None:
                        b.append(blurI[i,j])
                    else:
                        b.append(img[i,j])
            a.append(b)
        a=np.array(a)
        img = a
        return img


def create_folder(name):
    try:
        if not os.path.exists(name):
            os.makedirs(name)
    except OSError:
        print ('Error: Creating directory. ' + name)
        
            

# # 5. Print model Predict
# topk = 24
# num = 0
# image = np.array(image)
# xai_image = image
# top1_list = []

# for j in range(5):
#     result_path = save_path + '/top-'+str(j+1)
#     im = Image.fromarray(image)
#     blurI = im.filter(ImageFilter.BoxBlur(3))
#     try:
#         if not os.path.exists(result_path):
#             os.makedirs(result_path)
#     except OSError:
#         print ('Error: Creating directory. ' + result_path)
        
#     # 5, 6, 7
#     for k in tqdm.tqdm(range(10)):
#         score, label = obj.predict_image(xai_image, topk = topk)
#         dict_prob = {}
        
#         for i in range(topk):
#             dict_prob[str(obj.class_names[label[i] + 1])] = score[i].item()
#         max_label = max(dict_prob, key = dict_prob.get)
#         top1_list.append((max_label, dict_prob[max_label]))
        
#         sub_result_path = result_path + '/' + str(k) + '-(' + max_label + ', ' + str(dict_prob[max_label]) + ')'
#         try:
#             if not os.path.exists(sub_result_path):
#                 os.makedirs(sub_result_path)
#         except OSError:
#             print ('Error: Creating directory. ' + sub_result_path)
        
#         im = Image.fromarray(xai_image)
#         im.save(sub_result_path+'/image.png')
#         dict_prob_file = open(sub_result_path + '/dict_prob.txt', 'w')

#         for key,value in dict_prob.items():
#             dict_prob_file.write("{0} - Prob:{1:0.4f} \n".format(key, value))
#         dict_prob_file.close()
        
#         t,m = obj.show_explain(xai_image, feature_num = topk)
    
#         # 6. Positive Masking - Change Label
#         xai_image, mask = get_lime_mask(xai_image, blurI, m[0], mode = 1, blur = True)
        
#         exp = Image.fromarray(t[num])
#         exp.save(sub_result_path+'/explain.png')
#         clear_output(wait=True)
        
#     image, mask = get_lime_mask(image, blurI, m[0], mode = -1, blur = False)
#     xai_image = image
#     im = Image.fromarray(image)
#     im.save(result_path+'/image.png')