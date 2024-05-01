import pandas as pd
import glob
import sys
import torch
from torch.utils.data import Dataset,DataLoader
sys.path.append("/content/M22AIE212_SpeechUnderstanding_Assignment3")
sys.path.append("/content/SSL_Anti-spoofing")
from utils import convert_to_flac,Arguments,produce_evaluation_file,compute_eer,plot_roc_curve_with_auc
from utils import CustomDataset
from model import Model
from model_loader import ModelLoader
from train import train_epoch
from eval import evaluate_accuracy
import os

# Training and validation of LA for funtuning
batch_size = 2
num_epochs = 10
model_save_path = '/content/drive/MyDrive/Assignments/SpeechUnderstanding/A3/'

#set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Arguments
args = Arguments(
    la_model_path='/content/drive/MyDrive/Assignments/SpeechUnderstanding/A3/LA_model.pth',
    df_model_path='/content/drive/MyDrive/Assignments/SpeechUnderstanding/A3/Best_LA_model_for_DF.pth',
    eval=True,
    la_eval_output='/content/finetuned_la_score.txt',
    df_eval_output='/content/finetuned_df_score.txt'
)

df = pd.DataFrame(glob.glob("/content/for-2seconds/*/*/*.wav"), columns = ['file_path'])
df['real_or_fake'] = df['file_path'].apply(lambda x : x.split('/')[-2])
df['split_type'] = df['file_path'].apply(lambda x : x.split('/')[-3])
df['label'] = df['real_or_fake'].apply(lambda x : 1 if x=='real' else 0)
df = df.sample(len(df)) 

train_set = CustomDataset(df[df.split_type == 'training']['file_path'].tolist()[:2000],
                        df[df.split_type == 'training']['label'].tolist()[:2000])
test_set = CustomDataset(df[df.split_type == 'testing']['file_path'].tolist()[:400],
                        df[df.split_type == 'testing']['label'].tolist()[:400])
validation_set = CustomDataset(df[df.split_type == 'validation']['file_path'].tolist()[:400],
                        df[df.split_type == 'validation']['label'].tolist()[:400])


model_loader = ModelLoader(args, device)
model = model_loader.load_model('la')

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=False)
dev_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, drop_last=False)

## Training - LA
best_validation_loss = float('inf')
for epoch in range(num_epochs):
    running_loss = train_epoch(train_loader, model, args.lr, optimizer, device)
    validation_loss = evaluate_accuracy(dev_loader, model, device)
    print("Epoch: ", epoch, '\t', 'Train Loss: ', running_loss, '\t', 'Validation Loss: ', validation_loss)
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_finetuned_LA_model.pth'))

## Inference - LA
batch_size = 2
##inference using LA model, & saving scores in txt files
produce_evaluation_file(test_set,batch_size, model, device, args.la_eval_output)

la_df = pd.read_csv('/content/finetuned_la_score.txt', sep = ' ', header = None)
la_df.columns = ['actual', 'scores']

la_eer = compute_eer(la_df.actual, la_df.scores)
print("EER (Equal Error Rate) for LA model : ", round(la_eer, 4))

#plotting roc cureve with auc for LA model
plot_roc_curve_with_auc(la_df.actual, la_df.scores, 'la')
