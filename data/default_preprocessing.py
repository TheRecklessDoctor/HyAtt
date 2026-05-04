#####################################
# Modified by Dayan Perera
# Date: 26/06/2025
# Purpose: Preprocessing base class.
####################################

from pathlib import Path
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

class DefaultProcessor(ABC):

    """Preprocessor base class"""

    def __init__(
            self, 
            input_dir:str, 
            output_dir:str,
            dataset: Optional[str] = None
    ):

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dataset = dataset
        if self.dataset == "isles-2022":
            self.search_dir = Path(input_dir) 
        else:
            self.search_dir = Path(input_dir) / "raw_data"
        # self.base_dir = base_dir
        # self.rawdata_dir = self.base_dir + "raw_data"
        # self.derivatives_dir = self.base_dir + "derivatives"

        self.ct_types = ["ncct", "cta"]
        self.mri_types = ["dwi", "adc"]


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
    
    @abstractmethod
    def process_subject_session(
        self, 
        subject_id:str, 
        session_id:str
    ) -> bool:
        """This function contains code to preprocess a particular subject or session."""
        pass
    
    @abstractmethod
    def run_pipeline(
        self,
        subjects: Optional[List[str]] = None, 
        sessions: Optional[List[str]] = None
    ) -> None:
        """This function contains code that runs the entire preprocessing pipeline."""
        pass