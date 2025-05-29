
import torch
import pandas as pd
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from huggingface_hub import login
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, ViTModel
#from macenko import macenko_normalizer
#from vision_transformer import VisionTransformer    
import torch.nn as nn
from PIL import Image

def huggingfaceLogin():
    #must use a read only token
    #login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    login(token="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", add_to_git_credential=False)
    

'''
    Main entry point for using foundation models to get image features
'''
def extractFeatures(model_name, patches):

    feature_extractors = {
        'uni': extractFeaturesUsingUNI,
        'gigapath': extractFeaturesUsingGigaPath,
        'virchow': extractFeaturesUsingVirchow,
        'phikon': extractFeaturesUsingPhikon
    }
    
    # Convert the model name to lowercase
    model_name = model_name.lower()
    extractor_function = feature_extractors.get(model_name)
    
    # If the function exists, call it with patches, otherwise return None
    if extractor_function:
        return extractor_function(patches)
    else:
        return None

    


def getFeatureMap(patch):
    model = timm.create_model('resnet50d',pretrained=True, features_only=True)
    #image = transformImage(patch, model)
    image = torch.as_tensor(np.array(patch,dtype=np.float32)).transpose(2,0)[None]
    #print(image.shape)
    model.eval()

    with torch.inference_mode():
        feat_layers = model(image)
    return feat_layers
    
def mapToVec(feature_map):
    # Process each feature map to create a 1D feature vector
    # Option A: Global Average Pooling and then flattening
    #features = [F.adaptive_avg_pool2d(fmap, (1, 1)).view(fmap.size(0), -1) for fmap in feature_map]

    # Option B: Flatten the entire feature map directly (if you prefer not to pool)
    features = [fmap.view(fmap.size(0), -1) for fmap in feature_map]

    # Step 4: Concatenate all feature vectors from different layers
    feature_vector = torch.cat(features, dim=1)

    # `feature_vector` is now a 1D feature vector for the input image
    return feature_vector

def transformImage(image, model):
    # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    #print(transform)
    image = transform(image).unsqueeze(dim=0) 
    return image

def transformImages(images, model):
    # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    #print(transform)

    if isinstance(images[0], np.ndarray):
        # Convert each NumPy array back to a PIL Image
        images = [Image.fromarray(img) for img in images]

    transformed = [transform(image).unsqueeze(dim=0) for image in images ]
    return transformed

##################### FOUNDATION MODELS ########################################

'''
    GIGAPATH by Microsoft
'''
def extractFeaturesUsingGigaPath(patches):
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    transformed_images = transformImages(patches, model)
    # Stack the transformed images into a batch
    image_batch = torch.cat(transformed_images, dim=0) 

    model.eval()
    with torch.inference_mode():
        features = model(image_batch)
    
    return features.detach().cpu().numpy()

'''
    UNI by Mahmood Lab
'''
def extractFeaturesUsingUNI(patches):
    # pretrained=True needed to load UNI weights (and download weights for the first time)
    # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    
    transformed_images = transformImages(patches, model)
    # Stack the transformed images into a batch
    image_batch = torch.cat(transformed_images, dim=0) 
    model.eval()

    with torch.inference_mode():
        features = model(image_batch)
    
    # When you return feature_vector.detach().cpu().numpy(), 
    # you're converting the tensor to a NumPy array and moving it to the CPU.
    return features.detach().cpu().numpy()

'''
    VIRCHOW by Paige
'''
def extractFeaturesUsingVirchow(patches):
    # need to specify MLP layer and activation function for proper init
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

    transformed_images = transformImages(patches, model)

    # Stack the transformed images into a batch
    image_batch = torch.cat(transformed_images, dim=0) 
    model.eval()

    with torch.inference_mode():
         # Extracted features (torch.Tensor) with shape [1,1024]
        output = model(image_batch)

    '''
    from https://huggingface.co/paige-ai/Virchow2:
        We concatenate the class token and the mean patch token 
        to create the final tile embedding. 
        In more resource constrained settings, one can experiment 
        with using just class token or the mean patch token. 
        For downstream tasks with dense outputs (i.e. segmentation), 
        the 256 x 1280 tensor of patch tokens can be used.
    '''
    class_token = output[:, 0]    # size: 1 x 1280
    patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
    features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

    #return feature_vector.squeeze().detach().numpy()
    return features.detach().cpu().numpy()

'''
    PHIKON
'''
def extractFeaturesUsingPhikon(patches):
    # load phikon
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

    #image_batch = torch.cat(transformed_images, dim=0) 
    # process the image
    image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
    image_batch = image_processor(patches, return_tensors="pt")

    model.eval()
    with torch.inference_mode():
        outputs = model(**image_batch)
        features = outputs.last_hidden_state[:, 0, :]  # (1, 768) shape
    
    return features.detach().cpu().numpy()



##################### SINGLE PATCH ########################################

def extractFeatureVectorUsingUNI(patch):
    # pretrained=True needed to load UNI weights (and download weights for the first time)
    # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    image = transformImage(patch, model)
    model.eval()

    with torch.inference_mode():
        feature_vector = model(image) # Extracted features (torch.Tensor) with shape [1,1024]

    return feature_vector.squeeze().detach().numpy()



def extractFeatureVectorUsingGigaPath(patch):
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    image = transformImage(patch, model)

    model.eval()
    with torch.inference_mode():
        feature_vector = model(image)
    
    print(feature_vector)

    return feature_vector.squeeze().detach().numpy()


##################### NOT IN USE ########################################
def extractFeaturesUsingResNet50(patch, layer_ind=0):
    model = timm.create_model('resnet50d',pretrained=True, features_only=True)
    #image = transformImage(patch, model)
    image = torch.as_tensor(np.array(patch,dtype=np.float32)).transpose(2,0)[None]
    #print(image.shape)
    model.eval()

    with torch.inference_mode():
        feat_layers = model(image)
    
    #print(len(feat_layers))
    #last map of last layer
    #for layer in feat_layers:
    #    print(layer.shape)
    feature_vector = mapToVec(feat_layers[layer_ind][:,-1:,:,:])
    return feature_vector.squeeze().detach().numpy()








# def extractFeaturesUsingEXAONEPath(patch):
#
#     model = VisionTransformer.from_pretrained("LGAI-EXAONE/EXAONEPath", use_auth_token=hf_token)
#     transform = transforms.Compose(
#         [
#             transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
#             transforms.CenterCrop(224),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ]
#     )
#     normalizer = macenko_normalizer()
#     image_macenko = normalizer(patch)
#     sample_input = transform(image_macenko).unsqueeze(0)
#     model.cuda()
#     model.eval()
#     with torch.inference_mode():
#         features = model(sample_input.cuda())
#     return features
