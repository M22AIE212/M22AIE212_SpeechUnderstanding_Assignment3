import os
import glob
import subprocess

def convert_to_flac(input_dir):
    # Loop through files in the input directory
    for file_path in glob.glob(input_dir + "/*/*"):
        print(file_path)
        current_file = file_path
        new_file = '_'.join(file_path.strip().split()).replace('(', '').replace(')', '')
        
        # Rename file if it's not already in .flac format
        if '.flac' not in new_file:
            os.rename(current_file, new_file)
            flac_file = os.path.splitext(new_file)[0] + '.flac'
            print(flac_file)
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', new_file, '-ar', '16000', flac_file])
