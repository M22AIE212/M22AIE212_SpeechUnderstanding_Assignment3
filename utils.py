import os
import glob
import subprocess
from dataclasses import dataclass
from torch.utils.data import Dataset,DataLoader

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

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class Dataset_eval(Dataset):
  def __init__(self, file_path, label):
    self.file_path = file_path
    self.cut=64600//2 # take ~4//2 sec audio (64600//2 samples) ie 2 secs audio (32300 samples)
    self.label  = label
  def __len__(self):
    return len(self.file_path)
  def __getitem__(self, index):
    X, fs = librosa.load(self.file_path[index], sr=16000)
    X_pad = pad(X,self.cut)
    x_inp = Tensor(X_pad)
    label = self.label[index]
    return x_inp, label

@dataclass
class Arguments:
    la_model_path: str
    df_model_path: str
    eval: bool = True
    la_eval_output: str = '/content/la_score.txt'
    df_eval_output: str = '/content/df_score.txt'


