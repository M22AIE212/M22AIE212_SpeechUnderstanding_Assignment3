import sys
sys.path.append("/content/SSL_Anti-spoofing/")
sys.path.append("/content/M22AIE212_SpeechUnderstanding_Assignment3/")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch import Tensor
import librosa

from utils import convert_to_flac ,CustomDataset,Arguments,produce_evaluation_file,compute_eer,plot_roc_curve_with_auc
import glob
from model_loader import ModelLoader

if __name__ == "__main__" :
  ## set device
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  batch_size = 2

  ## Arguments
  args = Arguments(
      la_model_path='/content/drive/MyDrive/Speech Understanding (1)/A3/LA_model.pth',
      df_model_path='/content/drive/MyDrive/Speech Understanding (1)/A3/Best_LA_model_for_DF.pth',
      eval=True,
      la_eval_output='/content/la_score.txt',
      df_eval_output='/content/df_score.txt'
  )

  # ## Data Conversion
  input_directory = "/content/data/Dataset_Speech_Assignment"
  convert_to_flac(input_directory)


  ## Create a DataFrame with audio file paths
  df = pd.DataFrame(glob.glob("/content/data/Dataset_Speech_Assignment/*/*"), columns=['file_path'])

  ## Extract labels from file paths
  df['real_or_fake'] = df['file_path'].apply(lambda x: x.split('/')[-2])

  ## Assign labels (1 for 'Real', 0 for 'Fake')
  df['label'] = df['real_or_fake'].apply(lambda x: 1 if x == 'Real' else 0)


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model_loader = ModelLoader(args, device)

  # Evaluation Dataset
  eval_set = CustomDataset(df.file_path.tolist(), df.label.tolist())

  ## Inference - LA model
  model = model_loader.load_model('la')
  produce_evaluation_file(eval_set,batch_size, model, device, args.la_eval_output)

  ## Inference - DF model
  model = model_loader.load_model('df')
  produce_evaluation_file(eval_set,batch_size, model, device, args.df_eval_output)

  ## Evaluation - LA model
  la_df = pd.read_csv('/content/la_score.txt', sep = ' ', header = None)
  la_df.columns = ['actual', 'scores']
  la_eer = compute_eer(la_df.actual, la_df.scores)
  print("Equal Error Rate = LA model : ", round(la_eer, 4))
  plot_roc_curve_with_auc(la_df.actual, la_df.scores, 'la')

  ## Evaluation - DF model
  df_df = pd.read_csv('/content/df_score.txt', sep = ' ', header = None)
  df_df.columns = ['truth', 'scores']
  df_eer = compute_eer(df_df.truth, df_df.scores)
  print("Equal Error Rate = DF model : ", round(df_eer, 4))
  plot_roc_curve_with_auc(df_df.truth, df_df.scores, 'df')
