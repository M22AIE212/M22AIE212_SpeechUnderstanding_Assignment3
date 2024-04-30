
import os
import glob
import subprocess
from dataclasses import dataclass
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import det_curve,RocCurveDisplay,auc,roc_curve
import librosa
from torch import Tensor

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

def produce_evaluation_file(dataset,batch_size, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()

    for batch_x,label in data_loader:
      label_list = []
      score_list = []
      batch_size = batch_x.size(0)
      batch_x = batch_x.to(device)
      batch_out = model(batch_x)
      batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
      # add outputs
      label_list.extend(label)
      score_list.extend(batch_score.tolist())
      with open(save_path, 'a+') as fh:
          for l, s in zip(label_list,score_list):
              fh.write('{} {}\n'.format(l, s))
      fh.close()
    print('Scores are saved to {}'.format(save_path))

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


## EER
def compute_eer(truth, scores):
  frr, far, th = det_curve(truth, scores)
  abs_diffs = np.abs(frr - far)
  min_index = np.argmin(abs_diffs)
  eer = np.mean((frr[min_index], far[min_index]))
  return eer


## ROC - AUC
def plot_roc_curve_with_auc(truth, scores, la_or_df):
  fpr, tpr, thresholds = roc_curve(truth,scores)
  roc_auc = auc(fpr, tpr)
  display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
  display.plot()
  if la_or_df =='la':
    plt.title("ROC curve with AUC score for LA model by M22AIE227")
  else:
    plt.title("ROC curve with AUC score for DF model by M22AIE227")
  plt.show()
