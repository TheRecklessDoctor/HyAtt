#####################################
# Created by Dayan Perera
# Date: 26/06/2025
# Purpose: First half of the preprocessing including registration
# and skull stripping. However not in use as I pivoted to using MRI scans as input.
# However, have to test NCCT and CTA clipped as input once my model design is finalised and tested.
####################################

import logging
import sys
from typing import List, Optional
import argparse
import shutil
from pathlib import Path
import subprocess
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from default_preprocessing import DefaultProcessor #if I fix this error there is a module not found error that shows
# up and I am too lazy to fix it now. 

logging.basicConfig(
    level=logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stroke-processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ImageProcessor(DefaultProcessor):

    def __init__(
            self, 
            input_dir:str, 
            output_dir:str
    ):

        super().__init__(input_dir, output_dir)

    
    def find_image(
            self, 
            subject_id: str, 
            session_id: str, 
            mod:str
    ) -> Optional[str]:
        """
        Find a particular modality given a subject and session
        
        Args:
            subject_id (str): Subject identifier
            session_id (str): Session identifier
            mod (str): Modality to use as reference
            
        Returns:
            Path to DWI reference image or None if not found
        """
        # Look for mod image in the session directory
        session_dir = self.search_dir + f"/{subject_id}/{session_id}"
        
        # Check for mod file
        mod_pattern = f"{subject_id}_{session_id}_{mod}.nii.gz"
        mod_files = list(session_dir.glob(mod_pattern))
        
        if mod_files:
            # logger.info(f"Found {mod} reference: {mod_files[0]}")
            print(f"Found {mod} reference: {mod_files[0]}")
            return mod_files[0]
        
    
    def run_ants_registration(
            self, 
            fixed_image: Path, 
            moving_image:Path, 
            output_prefix:Path,
            is_label: bool = False
    ) ->bool:
        """
        Register the moving image to the fixed image.

        Args:
            fixed_image (Path): path of image to act as a baseline in registration
            moving_image (Path): path of image to register to fixed image.
            output_prefix (Path): file path name
            is_label (bool): boolean value to check if the file is the ground truth
        
        Return:
            boolean: True is successful and False otherwise
        """
        
        output_prefix.parent.mkdir(parents=True, exist_ok=True)

        if is_label: 
            set_interpolation = "NearestNeighbor"
            hist_matching = 0
        else:
            set_interpolation = "Linear"
            hist_matching = 1
        cmd = [
                "antsRegistration",
                "--dimensionality", "3",
                "--float", "0",
                "--output", f"[{output_prefix},1,1]",
                "--interpolation", f"{set_interpolation}",
                "--winsorize-image-intensities", "[0.005,0.995]",
                "--use-histogram-matching", f"{hist_matching}",
                
                # Initial moving transform
                "--initial-moving-transform", f"[{fixed_image},{moving_image},1]",
                
                # Rigid registration
                "--transform", "Rigid[0.1]",
                "--metric", f"MI[{fixed_image},{moving_image},1,32,Regular,0.25]",
                "--convergence", "[1000x500x250x100,1e-6,10]",
                "--shrink-factors", "8x4x2x1",
                "--smoothing-sigmas", "3x2x1x0vox",
                
                # Affine registration
                "--transform", "Affine[0.1]",
                "--metric", f"MI[{fixed_image},{moving_image},1,32,Regular,0.25]",
                "--convergence", "[1000x500x250x100,1e-6,10]",
                "--shrink-factors", "8x4x2x1",
                "--smoothing-sigmas", "3x2x1x0vox"
            ]
        
        logger.info("Running ANTS registration: %s -> %s", moving_image.name, fixed_image.name)
        try:
            _ = subprocess.run(cmd,text=True, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"{repr(e.stderr)}")

        transform_file = Path(str(output_prefix) + "0GenericAffine.mat")
        if transform_file.exists():
            logger.info("Registration completed: %s", transform_file)
            return True
        else:
            logger.info("Registration failed.")
            return False

    def run_hdbet_skull_stripping(
            self, 
            input_path: str, 
            output_path: str, 
            device: str = "cpu"
    ) -> bool:
        """
        Run HD-BET for skull stripping.
        
        Args:
            input_path (str): Input image path
            output_path (str): Output skull-stripped image path
            device (str): Device to use ('cpu' or 'cuda')
            
        Returns:
            bool: Success status
        """
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # HD-BET command
        cmd = [
            "hd-bet",
            "-i", str(input_path),
            "-o", str(output_path),  # Remove .nii.gz
            "-device", device,
            # "-mode", "fast",  # Use fast mode for efficiency
            "--disable_tta", # Disable test time augmentation for speed
            "--save_bet_mask" # save masks
        ]
        
        logger.info(f"Running HD-BET: {' '.join(cmd)}")
        _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # HD-BET creates _bet.nii.gz file, rename to expected output
        # hdbet_output = output_path + "_bet.nii.gz"
        if Path(output_path).exists():
            # hdbet_output.rename(output_path)
            logger.info("Skull stripping completed: %s", output_path)
            return True
        else:
            print("failed")
            logger.error("HD-BET output not found: %s", output_path)
            return False
                
        
    def apply_mask_to_image(self, image_path: Path, mask_path: Path, output_path: Path) -> bool:
        """
        Apply brain mask to skull-strip an image.
        
        Args:
            image_path (Path): Input image to mask
            mask_path (Path): Brain mask
            output_path (Path): Output masked image
            
        Returns:
            bool: Success status
        """
        # Load image and mask
        img = nib.load(image_path)
        mask = nib.load(mask_path)

        assert img.shape == mask.shape, f"shape mismatch {img.shape} vs {mask.shape}"
        
        # Apply mask
        img_data = img.get_fdata()
        mask_data = mask.get_fdata()
        
        # Ensure mask is binary
        mask_data = (mask_data > 0).astype(np.float32)
        
        # Apply mask
        masked_data = img_data * mask_data
        
        # Save masked image
        masked_img = nib.Nifti1Image(masked_data, img.affine, img.header)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(masked_img, output_path)
        
        logger.info("Mask applied: %s", output_path)
        return True
    
    def check_image_alignment(
            self, 
            img_path: Path, 
            ref_path: Path
    ) -> None:
        """
        Assert that two images have matching shape and affine.
        Raises AssertionError if mismatch.
        """
        img = nib.load(img_path).get_fdata()
        ref = nib.load(ref_path).get_fdata()
        logger.info(f"{img.shape} and {ref.shape}")

        assert img.shape == ref.shape, f"Shape mismatch: {img_path} {img.shape} vs {ref_path} {ref.shape}"
        # assert np.allclose(img.affine, ref.affine), f"Affine mismatch: {img_path} vs {ref_path}"
    
    def read_and_get_array(
            self,
            path: str
    ) -> list[sitk.Image, np.ndarray]:
        
        """
        Helper function to read the scan and return a numpy array version of the scan.

        Args:
            path(str): path to scan
        
        Return:
            [img, img_data](list[sitkImage, numpy array]): returns the read scan and the numpy array version.
        """
        
        img = sitk.ReadImage(path)
        img_data = sitk.GetArrayFromImage(img)

        return [img, img_data]


    
    def set_ods(
            self,
            img: np.ndarray,
            vals: list[tuple[float, float, float], tuple[float, float, float], tuple[float, ...]]
    ) -> np.ndarray:
        """
        Sets origin, direction and spacing for the given scan.

        Args:
            img: scan
            vals: [origin, spacing, direction]

        Returns:
            The scan although the nature of the library simply modifies the scans in place.
        """

        img.SetOrigin(vals[0])
        img.SetSpacing(vals[1])
        img.SetDirection(vals[2])

        return img



    
    def intensity_clipping(
            self,
            scans: dict[str, str],
            outs: dict[str, str],
            lower_bound: float = 0.0,
            upper_bound: float = 100.0
    ) -> None:
        
        """
        Perform intensity clipping on ncct and cta. 

        Args:
            scans(dict[str, str]): input paths
            outs(dict[str, str]): output paths
            lower_bound(float): lower bound to clip the intensity to.
            upper_bound(float): upper bound to clip the intensity to.
        
        Returns:
            None
        """

        ncct, ncct_data = self.read_and_get_array(scans["ncct"])
        _, cta_data = self.read_and_get_array(scans["cta"])
        # _, cbf_data = self.read_and_get_array(scans["cbf"])
        # _, cbv_data = self.read_and_get_array(scans["cbv"])
        # _, mtt_data = self.read_and_get_array(scans["mtt"])
        # _, tmax_data = self.read_and_get_array(scans["tmax"])

        origin = ncct.GetOrigin()
        direction = ncct.GetDirection()
        spacing = ncct.GetSpacing()

        ncct_clipped = np.clip(ncct_data, lower_bound, upper_bound)
        cta_clipped = np.clip(cta_data, lower_bound, upper_bound)
        # cbf_clipped = np.clip(cbf_data, 0, 35)
        # cbv_clipped = np.clip(cbv_data, 0, 10)
        # mtt_clipped = np.clip(mtt_data, 0, 20)
        # tmax_clipped = np.clip(tmax_data, 0, 7)

        ncct_clipped = sitk.GetImageFromArray(ncct_clipped)
        cta_clipped = sitk.GetImageFromArray(cta_clipped)
        # cbf_clipped = sitk.GetImageFromArray(cbf_clipped)
        # cbv_clipped = sitk.GetImageFromArray(cbv_clipped)
        # mtt_clipped = sitk.GetImageFromArray(mtt_clipped)
        # tmax_clipped = sitk.GetImageFromArray(tmax_clipped)

        ncct_clipped = self.set_ods(ncct_clipped, [origin, spacing, direction])
        cta_clipped = self.set_ods(cta_clipped, [origin, spacing, direction])
        # cbf_clipped = self.set_ods(cbf_clipped, [origin, spacing, direction])
        # cbv_clipped = self.set_ods(cbv_clipped, [origin, spacing, direction])
        # mtt_clipped = self.set_ods(mtt_clipped, [originspacing, direction])
        # tmax_clipped = self.set_ods(tmax_clipped, [origin, spacing, direction])

        # save clipped scans
        sitk.WriteImage(ncct_clipped, outs["ncct"])
        sitk.WriteImage(cta_clipped, outs["cta"])
        # sitk.WriteImage(cbv_clipped, outs["cbv"])
        # sitk.WriteImage(tmax_clipped, outs["cbf"])
        # sitk.WriteImage(mtt_clipped, outs["mtt"])
        # sitk.WriteImage(tmax_clipped, outs["tmax"])
        
        
    
    def process_subject_session(
            self, 
            subject_id: str,
            session_id: str = None
    ) -> bool:
                                
        """
        Process a single subject-session combination.
        
        Args:
            subject_id (str): Subject identifier
            session_id (str): Session identifier. However, this is used rarely and only if you want to process a specific
            session.
            
        Returns:
            bool: Success status
        """
        logger.info("Processing %s", subject_id)

        
        # Set up paths
        rawdata_subject = self.input_dir + "/raw_data/" + f"/{subject_id}"
        derivatives_subject = self.input_dir + "/derivatives/" + f"/{subject_id}"

        interim_subject = self.output_dir + f"/{subject_id}"
        Path(interim_subject).mkdir(parents=True, exist_ok=True)

        # I use the raw ncct, cta which is registered to the ncct and the skull-stripped dwi and adc
        inputs = {
            "ncct": rawdata_subject + f"/ses-01/{subject_id}_ses-01_ncct.nii.gz",
            "cta": derivatives_subject + f"/ses-01/{subject_id}_ses-01_space-ncct_cta.nii.gz",
            "dwi": rawdata_subject + f"/ses-02/{subject_id}_ses-02_dwi.nii.gz",
            "adc": rawdata_subject + f"/ses-02/{subject_id}_ses-02_adc.nii.gz",
            # "cbf": derivatives_subject + f"/ses-01/perfusion-maps/{subject_id}_ses-01_space-ncct_cbf.nii.gz",
            # "cbv": derivatives_subject + f"/ses-01/perfusion-maps/{subject_id}_ses-01_space-ncct_cbv.nii.gz",
            # "mtt": derivatives_subject + f"/ses-01/perfusion-maps/{subject_id}_ses-01_space-ncct_mtt.nii.gz",
            # "tmax": derivatives_subject + f"/ses-01/perfusion-maps/{subject_id}_ses-01_space-ncct_tmax.nii.gz",
            }

        outs = {
            "ncct": interim_subject + "/ncct_clipped.nii.gz",
            "cta" : interim_subject + "/cta_clipped.nii.gz",
            "dwi": interim_subject + "/dwi.nii.gz",
            "adc": interim_subject + "/adc.nii.gz",
            # "cbf": interim_subject + "/cbf.nii.gz",
            # "cbv": interim_subject + "/cbv.nii.gz",
            # "mtt": interim_subject + "/mtt.nii.gz",
            # "tmax": interim_subject + "/tmax.nii.gz"
        }

        print("Intensity clipping before")
        self.intensity_clipping(inputs, outs)
        
        
        shutil.copy(inputs["dwi"], interim_subject + "/")
        shutil.copy(inputs["adc"], interim_subject + "/")

        # # perform skull-stripping and it is not necesessary to skull strip the MRI scans as they are 
        # # already skull stripped so we simply copy the scans as per above.
        # ncct_skull_stripped = interim_subject + f"/{subject_id}_ncct_skull_stripped.nii.gz"
        # if not self.run_hdbet_skull_stripping(outs["ncct"], ncct_skull_stripped):
        #     logger.error("Failed to perform skull-stripping on ncct")
        #     return False
        
        # cta_skull_stripped = interim_subject + f"/{subject_id}_cta_skull_stripped.nii.gz"
        # if not self.run_hdbet_skull_stripping(outs["cta"], cta_skull_stripped):
        #     logger.error("Failed to perform skull-stripping on cta")
        #     return False
        

        
        return True   ,    
    
    def run_pipeline(
            self, 
            subjects: Optional[List[str]] = None, 
            sessions: Optional[List[str]] = None
    ) -> None:
        """
        Code for the complete processing pipeline.
        
        Args:
            subjects (List[str], optional): Specific subjects to process
            sessions (List[str], optional): Specific sessions to process. However, this option is rarely used and 
            only if you want to preprocess specific sessions. Be mindful as certain sessions only contain certain 
            modalities.
        """
        logger.info("Starting stroke imaging processing pipeline")
        
        # Find all subjects and sessions
        subjects_sessions = self.find_all_subjects_and_sessions()
        # print(subjects_sessions)
        
        total_processed = 0
        total_successful = 0
        
        for subject_id, _  in subjects_sessions.items():
            # Filter subjects if specified
            if subjects and subject_id not in subjects:
                continue
                
            total_processed += 1
            try:
                if self.process_subject_session(subject_id):
                    total_successful += 1
                    logger.info(f"Successfully processed {subject_id}")
                else:
                    logger.error(f"Failed to process {subject_id}")
            except Exception as e:
                logger.error(f"Error processing {subject_id}: {str(e)}")
        
        logger.info(f"Pipeline completed: {total_successful}/{total_processed} sessions processed successfully")

def main():

    parser = argparse.ArgumentParser("Ischemic stroke image processing pipeline")
    parser.add_argument("--mode", default="train", help="whether to process training or testing data")
    parser.add_argument("--base_dir", default="/home/dayan/Documents/PhD/Objective_1/data/raw", help="Raw data directory")
    parser.add_argument("--base_test_dir", default="/home/dayan/Documents/PhD/Objective_1/data/raw_test", help="directory of test data")
    parser.add_argument("--subjects", nargs="+", help="Specific subjects to process if you need")
    parser.add_argument("--sessions", nargs="+", help="Specific sessions to process if you need")
    parser.add_argument("--interim_dir", default="/home/dayan/Documents/PhD/Objective_1/data/interim", help="directory to store the interim preprocessed data")
    parser.add_argument("--interim_test_dir", default="/home/dayan/Documents/PhD/Objective_1/data/interim_test", help="directory of interim test data")
    args = parser.parse_args()

    if args.mode == "train":
        processor = ImageProcessor(args.base_dir, args.interim_dir)
        processor.run_pipeline(subjects=args.subjects, sessions=args.sessions)
    elif args.mode == "test":
        processor = ImageProcessor(args.base_test_dir, args.interim_test_dir)
        processor.run_pipeline(subjects=args.subjects, sessions=args.sessions)


if __name__ == "__main__":
    main()