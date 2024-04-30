import sys
sys.path.append("/content/SSL_Anti-spoofing/")

import torch
import torch.nn as nn
from model import Model ##imported from SSL_Anti-spoofing

class ModelLoader:
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def load_model(self, task):
        model = Model(self.args, self.device)
        total_parameters = sum([param.view(-1).size()[0] for param in model.parameters()])

        if task.lower() == 'la':
            model = model.to(self.device)
            print('Total parameters:', total_parameters)
            model.load_state_dict(torch.load(self.args.la_model_path, map_location=self.device))
            print('Model loaded:', self.args.la_model_path)
        else:
            model = nn.DataParallel(model).to(self.device)
            print('Total parameters:', total_parameters)
            model.load_state_dict(torch.load(self.args.df_model_path, map_location=self.device))
            print('Model loaded:', self.args.df_model_path)

        return model
