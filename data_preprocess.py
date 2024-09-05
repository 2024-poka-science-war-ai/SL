import os
from slp2memmap import Slp2MemmapConverter
from tqdm import tqdm
from multiprocessing import Pool

class SLPPreprocessor:
    def __init__(self, slp_base_path, memmap_base_path, num_workers=4):
        """Convert slp file to numpy memmap format

        Args:
            slp_base_path (str): base path of flattened slp files
            memmap_base_path (str): base path of converted memmap file
            num_workers (int): number of processes to use for parallel processing
        """
        self.slp_base_path = slp_base_path
        self.mm_base_path = memmap_base_path
        self.num_workers = num_workers
        self.converter = Slp2MemmapConverter()

    def _convert_file(self, args):
        """Helper function to convert a single file. Meant to be used with multiprocessing."""
        slp_file, mm_file, conf_file = args
        self.converter.convert(slp_file, mm_file, conf_file)
        return slp_file

    def process(self):
        """Convert all slp files to memmap format using multiprocessing"""
        file_names = [name.split(".")[0] 
                      for name in os.listdir(self.slp_base_path)]
        
        slp_abs_paths = [os.path.join(self.slp_base_path, f"{file_name}.slp") 
                         for file_name in file_names]
        
        mm_abs_paths = [os.path.join(self.mm_base_path, f"{file_name}.dat") 
                         for file_name in file_names]
        
        config_base_path = os.path.join(self.mm_base_path, 'config')
        os.makedirs(config_base_path, exist_ok=True)
        config_abs_paths = [os.path.join(config_base_path, f"{file_name}_conf.json")
                            for file_name in file_names]

        # Filter out files that have already been processed
        tasks = []
        for slp_file, mm_file, conf_file in zip(slp_abs_paths, mm_abs_paths, config_abs_paths):
            if not os.path.exists(mm_file):  # Skip if .dat file already exists
                tasks.append((slp_file, mm_file, conf_file))

        if not tasks:
            print("All files are already processed.")
            return

        pbar = tqdm(total=len(tasks))
        
        with Pool(processes=self.num_workers) as pool:
            for _ in pool.imap_unordered(self._convert_file, tasks):
                pbar.update(1)
                
        pbar.close()

def main(args):
    """Main function to convert slp to memmap, only used when running this file"""
    preprocessor = SLPPreprocessor(args.source_path, args.target_path, num_workers=args.num_workers)
    preprocessor.process()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="C:/Users/justi/Desktop/pk_ai/ssb/sl/ssbm-bot/training_data", help="source path of flattened slp dataset")
    parser.add_argument("--target_path", type=str, default="C:/Users/justi/Desktop/pk_ai/ssb/sl/SL-SL_develop/Data", help="target path of flattened memmap dataset")
    parser.add_argument("--num_workers", type=int, default=7, help="number of processes for parallel processing")
    args = parser.parse_args()
    main(args)