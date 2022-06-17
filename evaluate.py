# Author: Muhammed El-Yamani

import torch
from utils.scores import calc_accuracy
from tqdm import tqdm

# Get preds
def get_predictions(log_ps):
    with torch.no_grad():
        # get exp of log to get probabilities
        ps = torch.exp(log_ps)
        # get top_p and top_class
        top_p, top_class = ps.topk(1, dim=1)
        return top_class

# Make validation/test inference function
def evaluate(model, criterion, dataloader, desc='Validation'):
    """make validation or test inference based on the data"""

    model.eval()

    with torch.no_grad():
        # find what is the existed device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # intial helping variables
        num_batches = len(dataloader)
        accum_accuracy = 0
        running_loss = 0
        # iterate over the data
        for batch in tqdm(dataloader, total=num_batches, desc=f'{desc} round', unit='batch', leave=False):
            images, labels = batch
            labels, images = labels.to(device), images.to(device)
            # forward pass
            log_ps = model(images)
            # get predictions
            preds = get_predictions(log_ps).squeeze()
            # get loss
            loss = criterion(log_ps, labels)
            
            running_loss += loss.item()
            accum_accuracy += calc_accuracy(labels, preds)
            
        # get running_loss, accuracy metrics
        return running_loss / num_batches, accum_accuracy / num_batches