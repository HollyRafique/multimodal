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

DEBUG=True

class Aligner:
    def __init__(self, source_path, target_path, output_path, experiment_name="Align", temp_path="./tmp"):
        self.source_path = source_path
        self.target_path = target_path
        self.output_path = output_path
        self.temp_path = temp_path
        self.name = experiment_name
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
        registration_params : dict = deeperhistreg.configs.default_initial()
        #registration_params : dict = deeperhistreg.configs.default_initial_nonrigid()
        #registration_params : dict = deeperhistreg.configs.default_initial_fast()
        registration_params['loading_params']['source_resample_ratio']=1.0
        registration_params['loading_params']['target_resample_ratio']=1.0
        with open(os.path.join(self.temp_path,"params.json"), "w") as to_save:
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
        config['copy_target'] = True
        config['delete_temporary_results'] = False
        config['temporary_path'] = self.temp_path


        print(registration_params)
        ##### RUN REGISTRATION #####

        print("last poinnt before calling deeperhistreg")
        deeperhistreg.run_registration(**config)


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




def align(input_path='/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP',
          output_path= '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/AlignedH&E',
          #temp_path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/tmp',
          LEAPID="LEAP087",
          SLIDEID="slide 37",
          MAG="20x"
          ):

    ##### INITIALIZE ####

    LEVEL_OME=0
    LEVEL_NDPI=0
    #input_path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP' #/LEAP078.ome.tiff'
    #output_path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/AlignedH&E' #/LEAP078.ome.tiff'
    temp_path = '/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/tmp' #/LEAP078.ome.tiff'
    METHOD = 'deeperhistreg' ###'wsireg' #'deeperhistreg'
    #MAG="20x"

    #### READ FILES ####

    ome_tiff_files = glob.glob(os.path.join(input_path,"ConvertedTIFFs",MAG,"*tif*"))
    print(ome_tiff_files)
    #ome_tiff_files = glob.glob(os.path.join(input_path,"OME","*tif*"))
    target_path = [f for f in ome_tiff_files if LEAPID in f][0]
    print(f"target: {target_path}")


    #ndpi_files = glob.glob(os.path.join(input_path,"WSI","*.ndpi"))
    ndpi_files = glob.glob(os.path.join(input_path,"ConvertedH&Es","byLEAPID",f"{MAG}-tif","*.tif*"))
    print(ndpi_files)
    source_path = [f for f in ndpi_files if SLIDEID in f.lower()][0]
    print(f"source: {source_path}")

    # Get IF img from OME-TIFF
    #IF_img = tiff.imread(target_path)
    #print("IF image shape:", IF_img.shape)

    # Get H&E from ndpi

    #HE_slide = openslide.OpenSlide(source_path)
    #HE_img = HE_slide.read_region((0, 0), LEVEL_NDPI, HE_slide.level_dimensions[LEVEL_NDPI])
    #HE_img = HE_img.convert('RGB')
    #print("H&E image shape:", HE_img.size)


    aligner = Aligner(source_path, target_path, output_path,f"Align_{LEAPID}-{SLIDEID}", temp_path)

    method_name = f"align_with_{METHOD}"

    if hasattr(aligner, method_name):
        #call the method dynamically
        getattr(aligner, method_name)()
    else:
        print(f"Method {method_name} not found.")


    print("DONE")




if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-ip', '--input_path', required=True, help='path to input files')
    ap.add_argument('-op', '--save_path', required=True, help='path to save output')
    ap.add_argument('-tp', '--temp_path', help='path to store temporary files')
    ap.add_argument('-lid', '--leap_id',default='LEAP087', help='LEAPID')
    ap.add_argument('-sid', '--slide_id',default='slide 37', help='eg slide 37')
    ap.add_argument('-mag', '--mag', default='10x', help='eg 10x or 20x')
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

    print(f"ip: {args.input_path}")
    print(f"op: {save_path}")
    print(f"{args.leap_id} {args.slide_id} {args.mag}")

    align(args.input_path, save_path, args.leap_id, args.slide_id, args.mag)









