import torch
from .load_checkpoint import load_checkpoint
class EarlyStopping:
    """Save the best model during the trainning and finish trainning 
    if there is decrease for valid accuracy in delay epochs"""
    def __init__(self, delay, checkpoint_dict: dict, checkpoint_save="save.pth"):
        # path save chekpoint for the best model during training
        self.checkpoint_save = checkpoint_save
        # delay in number of epochs
        self.delay = delay
        # count continuous decrease in accuracy
        self.count_down = 0
        # record prev valid accuracy
        self.prev_valid_accuracy = None
        # record the best accuracy to save the best model
        self.best_accuracy = None
        
        self.checkpoint_dict = checkpoint_dict
    def track(self, valid_accuracy, model, train_loss, valid_loss, learning_rates_history, global_steps, current_epoch):
        self.model = model
        self.train_loss = train_loss 
        self.valid_loss = valid_loss
        self.valid_accuracy = valid_accuracy
        
        if self.prev_valid_accuracy != None and valid_accuracy <= self.prev_valid_accuracy:
            print("Warning: there is deacrease in valid accuracy")
            self.count_down += 1
        else:
            self.count_down = 0
        if self.best_accuracy == None or valid_accuracy > self.best_accuracy:
            print("Winning: better model")
            # save the best model
            self.checkpoint_dict['train_loss'] = train_loss
            self.checkpoint_dict['valid_loss'] = valid_loss
            self.checkpoint_dict['valid_accuracy'] = valid_accuracy
            self.checkpoint_dict['lr'] = learning_rates_history[-1]
            self.checkpoint_dict['global_steps'] = global_steps
            self.checkpoint_dict['epochs'] = current_epoch
            self.checkpoint_dict['state_dict'] = model.state_dict()

            torch.save(self.checkpoint_dict, self.checkpoint_save)
            # update the best accuracy metric
            self.best_accuracy = valid_accuracy
        # update prev_valid_accuracy
        self.prev_valid_accuracy = valid_accuracy
        if self.count_down == self.delay:
            # Finish training, there is continuous decreasing in accuracy
            return True
        else:
            return False
    def get_the_best_model(self):
        
        state_dict = torch.load(self.checkpoint_save)['state_dict']
        
        self.model.load_state_dict(state_dict)
        return self.model
    def measurements(self):
        return self.train_loss, self.valid_loss, self.valid_accuracy