import torch
import torch.nn as nn

def evaluate_accuracy(dev_loader, model, device):
  val_loss = 0.0
  num_total = 0.0
  model.eval()
  weight = torch.FloatTensor([0.1, 0.9]).to(device)
  criterion = nn.CrossEntropyLoss(weight=weight)
  for batch_x, batch_y in dev_loader:
    batch_size = batch_x.size(0)
    num_total += batch_size
    batch_x = batch_x.to(device)
    batch_y = batch_y.view(-1).type(torch.int64).to(device)
    batch_out = model(batch_x)

    batch_loss = criterion(batch_out, batch_y)
    val_loss += (batch_loss.item() * batch_size)
  val_loss /= num_total

  return val_loss
