import pandas as pd


class FileLoader:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.bf_df = None
        self.nuclei_df = None
        self.mit_df = None
        self._load_files()

    def _load_files(self):
        bf_files = [path for path in self.file_paths if 'BF' in path and path.endswith('.csv')]
        nuclei_files = [path for path in self.file_paths if 'Nuclei' in path and path.endswith('.csv')]
        mit_files = [path for path in self.file_paths if 'Mit' in path and path.endswith('.csv')]

        if len(bf_files) > 1:
            raise ValueError("应该只有一个唯一包含 'BF' 的 CSV 文件")
        elif len(bf_files) == 1:
            self.bf_df = pd.read_csv(bf_files[0])

        if len(nuclei_files) > 1:
            raise ValueError("应该只有一个唯一包含 'Nuclei' 的 CSV 文件")
        elif len(nuclei_files) == 1:
            self.nuclei_df = pd.read_csv(nuclei_files[0])

        if len(mit_files) > 1:
            raise ValueError("应该只有一个唯一包含 'Mit' 的 CSV 文件")
        elif len(mit_files) == 1:
            self.mit_df = pd.read_csv(mit_files[0])

