# Simplified DataLogger for a single CSV file
import csv
import os
import shutil
from datetime import datetime

class DataLogger:
    def __init__(self, logging_name, trajectory_name, headers):
        # Ensure log directory exists in the current working directory, with subdirectory for LOGGING_NAME and trajectory_name_datetime
        base_log_dir = os.path.join(os.getcwd(), 'logs')
        log_dir = os.path.join(base_log_dir, logging_name, trajectory_name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Compose log file name: always 'log.csv' in the log_dir
        self.csv_path = os.path.join(log_dir, 'log.csv')
        self.headers = headers
        self.file = open(self.csv_path, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.headers)
        self.closed = False

    def append_row(self, row):
        if len(row) != len(self.headers):
            raise ValueError(f"Row length {len(row)} does not match header length {len(self.headers)}")
        self.writer.writerow(row)

    def close(self):
        if not self.closed:
            self.file.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


    def copy_source_files_to_output(self, source_files_info):
        try:
            source_code_dir = os.path.join(self.output_folder, 'source_code')
            os.makedirs(source_code_dir, exist_ok=True)
            
            for filename, filepath in source_files_info.items():
                if os.path.exists(filepath):
                    shutil.copy2(filepath, os.path.join(source_code_dir, filename))
                    print(f"Copied {filename} to {source_code_dir}")
            
            print(f"Source code files copied to: {source_code_dir}")
            
        except Exception as e:
            print(f"Warning: Could not copy source files: {e}")