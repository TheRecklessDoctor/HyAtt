#####################################
# Adapted from: https://github.com/ycwu1997/CoactSeg/tree/Main
# Modified by Dayan Perera
####################################

import argparse
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import SimpleITK as sitk
import h5py

from default_preprocessing import DefaultProcessor



class Preprocessor():

    def __init__(
            self, 
            input_dir:str, 
            output_dir:str,
            dataset: str
    ):
        self.dataset = dataset
        if self.dataset == "isles-2022":
            self.search_dir = Path(input_dir)
        else:
            self.search_dir = Path(input_dir) / "raw_data"

        self.ground_truth_dir = input_dir  + "/derivatives/"
        self.input_dir = input_dir + "/raw_data/"
        self.output_dir = output_dir

    
    def find_all_subjects_and_sessions(self) -> Dict[str, List[str]]:
        """
        Finds all subjects and sessions in the directory.
        """

        print(f"Currently processing: {self.dataset}")
        print(f"Search path: {self.search_dir}")
        subjects_sessions = {}
        for subject_dir in self.search_dir.glob("sub-*"):
            if subject_dir.is_dir():
                subject_id = subject_dir.name
                # print(subject_id)
                sessions = []

                for session_dir in subject_dir.glob("ses-*"):
                    if session_dir.is_dir():
                        sessions.append(session_dir.name)
                
                if sessions:
                    subjects_sessions[subject_id] = sorted(sessions)
        
        print(f"Found {len(subjects_sessions)} subjects with sessions: {subjects_sessions}")
        return subjects_sessions
    

    def image_resample(
            self, 
            sitk_image: sitk.Image, 
            new_spacing: List[float] = None, 
            is_label:bool = False
    ) -> sitk.Image:
        
        """
        Parameters
        ----------
        new_spacing: currently the spacing to which to resample the image to is set to an isotrophic setting
        however, one can experiment with other values and even with anistrophic settings.
        """
        if new_spacing is None:
            new_spacing = [1.0, 1.0, 1.0]
        
        size = np.array(sitk_image.GetSize())
        spacing = np.array(sitk_image.GetSpacing())
        new_spacing = np.array(new_spacing)

        new_size = size * spacing / new_spacing
        new_spacing_refine = size * spacing / new_size
        new_spacing_refine = [float(s) for s in new_spacing_refine]
        new_size_refine = [int(s) for s in new_size]

        if not is_label:
            print(f"Original size: {size}")
            print(f"New size: {new_size_refine}")
            print(f"Original spacing: {spacing}")
            print(f"New spacing: {new_spacing_refine}")

        resample = sitk.ResampleImageFilter()
        resample.SetOutputDirection(sitk_image.GetDirection())
        resample.SetOutputOrigin(sitk_image.GetOrigin())
        resample.SetOutputSpacing(new_spacing_refine)
        resample.SetSize(new_size_refine)

        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)
        new_image = resample.Execute(sitk_image)

        return new_image
    

    def get_ods(self, scan_path:str) -> dict[str, sitk.VectorDouble]:

        """
        Gets origin, direction and spacing for the given scan.

        Args:
            scan_path (str): path to scan to get origin, direction and spacing values from.

        Returns:
            dict[str, sitk.VectorDouble]: dictionary with values for origin, direction and spacing.
        """

        scan = sitk.ReadImage(scan_path)

        origin = scan.GetOrigin()
        direction = scan.GetDirection()
        spacing = scan.GetSpacing()

        return {
            "origin": origin,
            "direction": direction,
            "spacing": spacing
        }
    
    def save_scan(
            self, 
            scan:np.array,
            scan_type: str, 
            stage:str,
            subject_id: int,
            metadata: dict[str, sitk.VectorDouble]
    ) -> sitk.Image:
        
        image = sitk.GetImageFromArray(scan)

        image.SetOrigin(metadata["origin"])
        image.SetDirection(metadata["direction"])
        image.SetSpacing(metadata["spacing"])

        save_path = self.output_dir + f"/{subject_id}/{scan_type}_{stage}.nii.gz"
        sitk.WriteImage(image, save_path)

        return image

    def process_single_scan(
            self,
            scan_type: str,
            scan_path: str,
            metadata: dict[str, sitk.VectorDouble],
            subject_id: str,
            verbose: Optional[bool] = False
    ) -> np.ndarray:
        
        """
        Function to preprocess a single scan

        Args:
            scan_type (str): type of scan
            scan_path (str): path to scan
            metadata (dict[str, sitk.VectorDouble]): dictionary containing values for origin, direction and spacing to set
                                                    for visualisation purposes.
            subject_id (str): subject id
            verbose (Optional[bool]): boolean value if you want to dispaly text during process i.e. for debugging/logging.
        
        Returns: 
            np.ndarray: numpy array version of preprocessed scan.
        """
        
        if verbose:
            print(f"Reading {scan_type} from {scan_path}")

        scan = sitk.ReadImage(scan_path)
        
        is_label = True if scan_type == "label" else False
        resampled = self.image_resample(
            sitk_image=scan, 
            is_label=is_label, 
            new_spacing=[1.0, 1.0, 1.0]
        )
        resampled_array = sitk.GetArrayFromImage(resampled)

        if is_label:
            print(f"min and max: {resampled_array.min()} and {resampled_array.max()}")
        else:
            print(f"Resampling complete on: {scan_type}")

        # perform z-score normalisation on everything but the labels
        if not is_label:
             resampled_array = (resampled_array - np.mean(resampled_array)) / np.std(resampled_array)
             print(f"Normalisation and preprocessing complete on: {scan_type}")
        else:
            print(f"Preprocessing complete on: {scan_type}")

        _ = self.save_scan(
            scan=resampled_array,
            scan_type=scan_type,
            stage="normalisation",
            subject_id=subject_id,
            metadata=metadata
        )

        return resampled_array

    
    def process_scans(
            self, 
            scans: dict[str, str], 
            subject_id: str
    ) -> dict[str, np.ndarray]:
        
        """
        Function to run preprocessing for a single subject.

        Args:
            scans (dict[str, str]): dictionary containing scan type and path for all scans for a single subject.
            subject_id (str): subject id
        
        Return:
            dict[str, np.ndarray]: dictionary containing scan type and numpy array version of preprocessed sample for
                                    all scans for a single subject.
        """
        
        samples = {}
        scan_type = "dwi"
        metadata = self.get_ods(scans[scan_type])

        for key, value in  scans.items():
            processed_sample = self.process_single_scan(
                scan_type = key, 
                scan_path = value, 
                metadata = metadata, 
                subject_id = subject_id,
                verbose = True
            )

            samples[key] = processed_sample
    
        return samples

    def process_subject_session(
            self, 
            subject_id: str
    ) -> bool:
        
        """
        Function to set up preprocessing for a single subject or a single session.

        Args:
            subject_id (str): id of the subject to process.
        
        Returns:
            bool: boolean representing success or failure. True if successful and False otherwise.
        """


        # setup paths
        raw_subject = self.input_dir + f"/{subject_id}/ses-02/"
        processed_subject = self.output_dir + f"/{subject_id}"

        # we need to create the output folder for this particular subject if it doesn't exist
        Path(processed_subject).mkdir(parents=True, exist_ok=True)

        # modalities
        if self.dataset == "isles-2024":
            dwi = raw_subject + f"{subject_id}_ses-02_dwi.nii.gz"
            adc = raw_subject + f"{subject_id}_ses-02_adc.nii.gz"
            ground_truth = self.ground_truth_dir + f"/{subject_id}"+"/ses-02"+ f"/{subject_id}_ses-02_lesion-msk.nii.gz"
            
            scans = {
                "dwi": dwi,
                "adc": adc,
                "label": ground_truth
            }
            
            scans = self.process_scans(
                scans = scans,
                subject_id = subject_id
            )

            # dwi_processed = scans["dwi"]
            # adc_processed = scans["adc"]
            # label_processed = scans["label"]

            file_name = processed_subject + f"/{subject_id}.h5"
            f = h5py.File((file_name), 'w')
            for key, item in scans.items():
                f.create_dataset(key, data=item, compression="gzip")
            # f.create_dataset('dwi', data=dwi_processed, compression="gzip")
            # f.create_dataset('adc', data=adc_processed, compression="gzip")
            # f.create_dataset('label', data=label_processed, compression="gzip")
            f.close()
        else:
            # this is the isles-2022 dataset
            dwi = raw_subject + f"/ses-0001/dwi/{subject_id}_ses-0001_dwi.nii.gz"
            adc = raw_subject + f"/ses-0001/dwi/{subject_id}_ses-0001_adc.nii.gz"
            ground_truth = self.base_dir + f"/derivatives/{subject_id}/ses-0001/{subject_id}_ses-0001_msk.nii.gz"
            scans = {
                "dwi": dwi,
                "adc": adc,
                "label": ground_truth
            }

            
            scans = self.process_scans(
                scans = scans,
                subject_id = subject_id
            )

            # dwi_processed = scans["dwi"]
            # adc_processed = scans["adc"]
            # label_processed = scans["label"]

            file_name = processed_subject + f"/{subject_id}.h5"
            # print(f"min and max: {label_processed.min()} and {label_processed.max()}")
            # print(f"Saving to : {file_name}")
            f = h5py.File(file_name, 'w')
            for key, item in scans.items():
                f.create_dataset(key, data=item, compression="gzip")
            # f.create_dataset("dwi", data=dwi_processed, compression="gzip")
            # f.create_dataset("adc", data=adc_processed, compression="gzip")
            # f.create_dataset("label", data=label_processed, compression="gzip")
            f.close()

        return True



        
    def run_pipeline(self) -> None:
        """
        Run the complete processing pipeline.
        
        """
        
        # Find all subjects and sessions
        subjects_sessions = self.find_all_subjects_and_sessions()
        
        total_processed = 0
        total_successful = 0
        success = None
        
        for subject_id, _  in subjects_sessions.items():
                
            try:
                success = self.process_subject_session(subject_id)
            except Exception as e:
                print(f"\nError processing {subject_id}: {str(e)}")
            if success:
                print(f"\n Successfully processed {subject_id}")
                total_successful += 1
            else:
                print(f"Unsuccessfully processed {subject_id}")
            
            total_processed += 1
        
        print(f"\nPipeline completed: {total_successful}/{total_processed} sessions processed successfully")

def main():

    parser = argparse.ArgumentParser("Ischemic stroke image processing pipeline")
    parser.add_argument("--mode", default="train", help="whether to process training or testing data")
    parser.add_argument("--base_dir", default="/home/dayan/Documents/PhD/Objective_1/data/raw", help="Raw data directory")
    parser.add_argument("--base_test_dir", default="/home/dayan/Documents/PhD/Objective_1/data/raw_test", help="directory of test data")
    parser.add_argument("--raw_dir", default="/home/dayan/Documents/PhD/Objective_1/data/raw", help="directory to store the interim preprocessed data")
    parser.add_argument("--raw_test_dir", default="/home/dayan/Documents/PhD/Objective_1/data/raw_test", help="directory of interim test data")
    parser.add_argument("--processed_dir", default="/home/dayan/Documents/PhD/Objective_1/data/processed", help="directory to store the final preprocessed data")
    parser.add_argument("--processed_test_dir", default="/home/dayan/Documents/PhD/Objective_1/data/processed_test", help="directory of final preprocessed test data")
    parser.add_argument("--dataset", default="isles-2024", help="which dataset to process")
    args = parser.parse_args()    

    if args.mode == "train":
        processor = Preprocessor(args.raw_dir, args.processed_dir, args.dataset)
        processor.run_pipeline()
    elif args.mode == "test":
        print(args.subjects)
        processor = Preprocessor(args.raw_test_dir, args.processed_test_dir, args.dataset)
        processor.run_pipeline()


if __name__ == "__main__":
    main()       
        