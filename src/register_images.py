'''
   Image registration using deeperhistreg

'''

import os
import ctypes
# Load the libvips library
vips_lib = ctypes.CDLL('/share/apps/vips-8.16.0/libvips/libvips.so.42')
# Load the OpenSlide library
openslide_lib = ctypes.CDLL('/share/apps/openslide-3.4.1/lib/libopenslide.so')
#Load tbb for wsireg
tbb_lib = ctypes.CDLL('/share/apps/onetbb-2021.1.1/lib64/libtbb.so.12')

import yaml
import datetime
import argparse
import openslide
import pyvips
import glob
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import exposure
import json
import re
import torch
from LEAPdata import get_slide_id, get_leap_ids

DEBUG=True

# Function to recursively update the dictionary
def update_dict_in_place(original, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and key in original:
            update_dict_in_place(original[key], value)  # Recursively update nested dictionaries
        else:
            original[key] = value  # Update/overwrite values

# Function to return a new dictionary with updates applied
def update_dict(original, updates):
    new_dict = original.copy()  # Create a shallow copy to preserve the original structure
    for key, value in updates.items():
        if isinstance(value, dict) and key in original:
            new_dict[key] = update_dict(original[key], value)  # Recursively update nested dictionaries
        else:
            new_dict[key] = value  # Update/overwrite values
    return new_dict


def extract_all_ids(filename):
    # Find all LEAP occurrences
    leap_pattern = r'LEAP\d+'
    leap_ids = re.findall(leap_pattern, filename, re.IGNORECASE)  # Returns a list of all matches

    # Extract Slide ID (first occurrence)
    slide_pattern = r'Slide[-_\s]?(\d+)'
    slide_match = re.search(slide_pattern, filename, re.IGNORECASE)

    slide_id = f"slide_{slide_match.group(1)}" if slide_match else None

    return leap_ids, slide_id


def extract_ids(filename):
    # This regex finds "LEAP" followed by digits, then later "Slide" (with an optional hyphen, underscore, or space) and digits.
    pattern = r'(LEAP\d+).*?Slide[-_\s]?(\d+)'
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        leap_id = match.group(1)
        slide_id = match.group(2)
        slide_id = f"slide_{slide_id}"
        return leap_id, slide_id
    
    ## if no match then try the WSI format

    pattern = r'(SLIDE_\d+)_(LEAP\d+)'
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        leap_id = match.group(2)
        #leap_id = f"LEAP{slide_id}"
        slide_id = match.group(1)
        #slide_id = f"slide_{slide_id}"
        return leap_id, slide_id
    else:
        return None, None



class Aligner:
    def __init__(self, source_path, target_path, output_path, experiment_name="Align", temp_path="./tmp", nonrigid=False, config=None):
        self.source_path = source_path
        self.target_path = target_path
        self.output_path = output_path
        self.temp_path = temp_path
        self.name = experiment_name
        self.nonrigid = nonrigid
        self.config = config
        os.makedirs(output_path,exist_ok=True)
        os.makedirs(os.path.join(output_path,self.name),exist_ok=True)


        print("Aligner initialised")

    def align_with_deeperhistreg(self):
        print("aligning with deeperhistreg")
        import deeperhistreg
        from deeperhistreg.dhr_input_output.dhr_loaders.tiff_loader import TIFFLoader
        from deeperhistreg.dhr_input_output.dhr_loaders.openslide_loader import OpenSlideLoader
        from deeperhistreg.dhr_pipeline.registration_params import default_initial, default_initial_nonrigid, default_initial_fast,default_initial_nonrigid_high_resolution
        ##### INIT REGISTRATION #####

        if self.nonrigid:
            registration_params : dict = deeperhistreg.configs.default_initial_nonrigid()
        else:
            registration_params : dict = deeperhistreg.configs.default_initial()


        if self.config:
            print('using config file')
            registration_params = update_dict(registration_params, self.config)
       
        registration_params['loading_params']['source_resample_ratio']=0.5
        registration_params['loading_params']['target_resample_ratio']=0.5
        registration_params['logging_path']=self.output_path

        ## These params should be set in a config file
        #registration_params['echo']=True
        #registration_params['initial_registration_params']['echo']=True
        #registration_params['initial_registration_params']['run_superpoint_ransac']=True
        #registration_params['initial_registration_params']['angle_step']=2
        #registration_params['initial_registration_params']['registration_size'] = 2048
        #registration_params['initial_registration_params']['registration_sizes'] = [50,100,550,600,650,700,750,800,850,900,1000,1250]


        with open(os.path.join(self.output_path,"reg_params.json"), "w") as to_save:
            json.dump(registration_params, to_save)
        # want to add date and time for the case_name

        #import json
        #with open("D:/04 MultiomicDatasets/LEAP/params.json", "w") as to_save:
        #    json.dump(registration_params, to_save)


        ## Create Config 
        config = dict()
        config['source_path'] = self.source_path
        config['target_path'] = self.target_path
        config['output_path'] = self.output_path
        config['registration_parameters'] = registration_params
        config['case_name'] = self.name
        config['save_displacement_field'] = True
        config['copy_target'] = False
        config['delete_temporary_results'] = False
        config['temporary_path'] = self.temp_path


        #print(registration_params)
        ##### RUN REGISTRATION #####

        print("calling deeperhistreg")
        deeperhistreg.run_registration(**config)
        #deeperhistreg.run_registration(**config).pipeline.initial_transform


        ##### APPLY DEFORMATION
        #displacement_field_path = os.path.join(self.output_path, "displacement_field.mha")
        #warped_path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/warped'
        #loader = deeperhistreg.loaders.TIFFLoader
        #saver = deeperhistreg.savers.TIFFSaver
        #save_params = deeperhistreg.savers.tiff_saver.default_params
        #level = 0 # Pyramid level to perform the warping - 0 is the highest possible
        #pad_value = 255 # White in the RGB representation
        #save_source_only = True # Whether to save only the warped image or also the corresponding target image
        #to_template_shape = True # Whether to align the source shape to template shape (if initially different)
        #to_save_target_path = None # Path where to save the target (if save_source_only set to False)

        #deeperhistreg.apply_deformation(
        #    source_image_path = self.source_path,
        #    target_image_path = self.target_path,
        #    warped_image_path = warped_path, 
        #    displacement_field_path = displacement_field_path,
        #    loader = loader,
        #    saver = saver,
        #    save_params = save_params,
        #    level = level,
        #    pad_value = pad_value,
        #    save_source_only = save_source_only,
        #    to_template_shape = to_template_shape,
        #    to_save_target_path = to_save_target_path
        #)  
       



    def align_with_tiatoolbox(self):
        print("aligning with tiatoolbox")
        print("TODO")

    def align_with_wsireg(self):
        print("aligning with wsireg")
        from wsireg.wsireg2d import WsiReg2D
        import itk

        # initialize registration graph
        reg_graph = WsiReg2D(self.name, os.path.join(self.output_path,self.name))

        # add registration images (modalities)
        reg_graph.add_modality(
            "modality_fluo",
            self.target_path,
            image_res=0.398,
            channel_names=["DNA", "PanCK", "CD45","Ki67"],
            channel_colors=["blue", "green", "red","yellow"],
            preprocessing={
                "image_type": "FL",
                "ch_indices": [0],
                "as_uint8": True,
                "contrast_enhance": True,
            },
        )

        reg_graph.add_modality(
            "modality_brightfield",
            self.source_path,
            image_res=0.8847,
            preprocessing={
                "image_type": "BF",
                "as_uint8": True,
                "invert_intensity": True,
            },
        )

        # we register here
        # using a rigid and affine parameter map
        reg_graph.add_reg_path(
            "modality_brightfield",
            "modality_fluo",
            thru_modality=None,
            reg_params=["affine"],
        )


        selx = itk.ElastixRegistrationMethod.New()
        selx.LogToConsoleOn()  # To print log to console


        print("****** ABOUT TO REGISTER IMAGES ****")
        # run the graph
        reg_graph.register_images()

        print("****** FINISHED REGISTERING IMAGES ****")
        # save transformation data
        reg_graph.save_transformations()

        # save registerd images as ome.tiff writing images
        # plane by plane
        reg_graph.transform_images(file_writer="ome.tiff")




#def align(input_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP',
#          output_path= '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/AlignedH&E',
#          #temp_path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/tmp',
#          LEAPID="LEAP087",
#          SLIDEID="slide 37",
#          MAG="20x"
#          ):
def align(source_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP',
          target_path= '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP',
          save_path= '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/AlignedH&E',
          #temp_path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/tmp',
          LEAPID="LEAP087",
          SLIDEID="slide 37",
          MAG="20x", NONRIGID=False, config=None
          ):

    ##### INITIALIZE ####

    LEVEL_OME=0
    LEVEL_NDPI=0
    #input_path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP' #/LEAP078.ome.tiff'
    #output_path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/AlignedH&E' #/LEAP078.ome.tiff'
    temp_path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/tmp' #/LEAP078.ome.tiff'
    METHOD = 'deeperhistreg' ###'wsireg' #'deeperhistreg'
    #MAG="20x"


    save_path = os.path.join(save_path,f"Align_{LEAPID}-{SLIDEID}")
    os.makedirs(save_path,exist_ok=True)
    print(f"config: {config}")
    aligner = Aligner(source_path, target_path, save_path,f"{LEAPID}_{SLIDEID}", temp_path, NONRIGID, config)

    method_name = f"align_with_{METHOD}"

    if hasattr(aligner, method_name):
        #call the method dynamically
        getattr(aligner, method_name)()
    else:
        print(f"Method {method_name} not found.")

    torch.cuda.empty_cache()

    print("DONE")


def align_multiple(input_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP',
          output_path= '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/AlignedH&E',
          #temp_path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/tmp',
          MAG="10x", nonrigid=False, config=None
          ):
    print("ALIGN MULTIPLE")
    print(f"config: {config}")
    #### READ FILES ####

    ome_tiff_files = sorted(glob.glob(os.path.join(input_path,"ConvertedTIFFs",MAG,"*tif*")))
    ndpi_files = sorted(glob.glob(os.path.join(input_path,"ConvertedHnEs","byLEAPID",f"{MAG}-tif","*.tif*")))

    for source_path in ndpi_files:

        LEAPID, SLIDEID = extract_ids(source_path)
        print(f"{LEAPID} and {SLIDEID}")
        
        target_path = [f for f in ome_tiff_files if LEAPID.lower() in f.lower()]

        if target_path:
            target_path = target_path[0]
        else:
            print(f"no file for {LEAPID}")
            continue

        print(f"source: {source_path}")
        print(f"target: {target_path}")
            
        align(source_path,target_path, output_path, LEAPID, SLIDEID, MAG, nonrigid,config)


    ##OLD METHOD
    ##for target_path in ome_tiff_files:
    ##    ndpi_files = glob.glob(os.path.join(input_path,"ConvertedHnEs","byLEAPID",f"{MAG}-tif","*.tif*"))
    ##    print(ndpi_files)
    ##    all_LEAPIDS, SLIDEID = extract_all_ids(target_path)
    ##    for LEAPID in all_LEAPIDS:
    ##        print(f"{LEAPID} and {SLIDEID}")
    ##        ndpi_paths = [f for f in ndpi_files if SLIDEID.lower() in f.lower()]
    ##        ndpi_paths = [f for f in ndpi_files if LEAPID.lower() in f.lower()]
    ##        if not ndpi_paths:
    ##            print(f"no file for {SLIDEID}")
    ##            continue
    ##        source_path = ndpi_paths[0]
    ##        print(f"source: {source_path}")
    ##        align(source_path,target_path, output_path, LEAPID, SLIDEID, MAG, nonrigid,config)


def align_single(input_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP',
          output_path= '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/AlignedH&E',
          #temp_path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/tmp',
          LEAPID="LEAP087",
          SLIDEID="slide_37",
          MAG="10x", nonrigid=False, config=None
          ):
    print("ALIGN SINGLE")
    #### READ FILES ####
    print(f"LEAPID: {LEAPID}")
    print(f"SLIDEID: {SLIDEID}")
    ome_tiff_files = glob.glob(os.path.join(input_path,"ConvertedTIFFs",MAG,"*tif*"))
    print(ome_tiff_files)
    #ome_tiff_files = glob.glob(os.path.join(input_path,"OME","*tif*"))
    target_path = [f for f in ome_tiff_files if LEAPID in f][0]
    print(f"\ntarget: {target_path}")


    #ndpi_files = glob.glob(os.path.join(input_path,"WSI","*.ndpi"))
    ndpi_files = glob.glob(os.path.join(input_path,"ConvertedHnEs","byLEAPID",f"{MAG}-tif","*.tif*"))
    print(ndpi_files)
    
    ndpi_files = [f for f in ndpi_files if SLIDEID.lower() in f.lower()]
    print(ndpi_files)
    source_path = [f for f in ndpi_files if LEAPID.lower() in f.lower()][0] 
    print(f"\nsource: {source_path}")

    align(source_path, target_path, output_path, LEAPID, SLIDEID, MAG, nonrigid, config)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-ip', '--input_path', required=True, help='path to input files')
    ap.add_argument('-op', '--save_path', required=True, help='path to save output')
    ap.add_argument('-tp', '--temp_path', help='path to store temporary files')
    ap.add_argument('-lid', '--leap_id', help='LEAPID')
    ap.add_argument('-sid', '--slide_id', help='eg slide 37')
    ap.add_argument('-mag', '--mag', default='10x', help='eg 10x or 20x')
    ap.add_argument('-m', '--multiple', action='store_true')
    ap.add_argument('-nr', '--nonrigid', action='store_true')
    ap.add_argument('-cp', '--config_path', help='full path to config file')


    args = ap.parse_args()

    #get current date and time for model name
    curr_date=str(datetime.date.today())
    curr_time=datetime.datetime.now().strftime('%H%M')

    #with open(args.config_file) as yaml_file:
    #    config=yaml.load(yaml_file, Loader=yaml.FullLoader)

    name=f"register_{curr_date}_{curr_time}"
    if DEBUG: print(name)
    #set up paths for models, training curves and predictions
    save_path = os.path.join(args.save_path,name)
    os.makedirs(save_path,exist_ok=True)

    # READ CONFIG
    if args.config_path:
        with open(args.config_path, "r") as file:
            config = yaml.safe_load(file)
    else:
        config = None



    print(f"ip: {args.input_path}")
    print(f"op: {save_path}")
    print(f"mag: {args.mag}")
    print(f"config: {config}")

    multiple=False
    if args.multiple:
        align_multiple(args.input_path, save_path, args.mag, args.nonrigid, config)
    else:

        # If slide ID provided then use it, otherwise look it up
        if args.slide_id:
            slide_id = args.slide_id
        else:
            slide_id = get_slide_id(args.leap_id)


        print(f"{args.leap_id} {slide_id}")
        align_single(args.input_path, save_path, args.leap_id, slide_id, args.mag, args.nonrigid, config)











