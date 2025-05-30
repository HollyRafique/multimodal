import os
import glob
import pandas as pd
from utils import save_features_with_names
from feature_extractor import extractFeatures
from huggingface_hub import login
import time
from PIL import Image

print(f"{time.strftime('%H:%M')} starting get_features",flush=True)         
#login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
login(token="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", add_to_git_credential=False)


model_names = ['uni', 'gigapath', 'virchow']
segments=['CD45+','PanCK+']

input_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/patches'
output_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/features'

print(f"{time.strftime('%H:%M')} about to start for loop",flush=True)         
for model in model_names:
    for segment in segments:
        print(f"{time.strftime('%H:%M')} model: {model} segment: {segment}",flush=True)         

        patch_paths = glob.glob(os.path.join(input_path,segment,'*.png'))
        image_names = [os.path.basename(path) for path in patch_paths]
        patch_metadata = pd.DataFrame([x.replace('.png','').split('_') for x in image_names],columns=['name', 'LEAP_ID', 'Segment','ROI_num', 'x', 'y'])

        
        patches = [Image.open(path) for path in patch_paths]
        feature_matrix = extractFeatures(model, patches)
        print(f"{time.strftime('%H:%M')} saving features",flush=True)         
        save_features_with_names(feature_matrix, patch_paths, output_path, f"{model}-{segment}", format="csv")
        
        
    
