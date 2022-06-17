# Author: Muhammed El-Yamani
import os
from numpy import load
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import re
import wandb
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import logging
from pathlib import Path

# helper files
from evaluate import evaluate, get_predictions
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils import utils
from .early_stopping import EarlyStopping


class Net_Utility:

    def __init__(self, net, checkpoint_dict: dict, optimizer, criterion, data_loaders: dict, image_datasets: dict, img_size: tuple, delay: int = 2,  n_classes: int = 46, save_checkpoint: bool = True, is_train: bool = True, type_load_model: str = 'load_best',  epochs: bool = 1, batch_size: int = 16, learning_rate: float = 1e-5, print_every: int = None, dir_checkpoint: str = None, load_model: str = False, debug=False):
        '''
        delay: int: number of ebochs delay to choose the best model in early stopping
        type_load_model: load_best | load_interrupted | load_last
        '''
        self.net = net
        self.checkpoint_dict = checkpoint_dict
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_loaders = data_loaders
        self.image_datasets = image_datasets
        self.img_size = img_size
        self.n_classes = n_classes
        self.save_checkpoint = save_checkpoint
        self.is_train = is_train
        self.type_load_model = type_load_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.print_every = print_every
        self.delay = delay
        self.debug = debug
        # default None print_very 10% epoch
        self.net_train_loss = None
        self.net_valid_loss = None
        self.net_valid_accuracy = None
        self.train_loss_history = []
        self.valid_loss_history = []
        self.valid_accuracy_history = []
        self.learning_rates_history = []
        self.steps_history = []
        self.current_epoch = 0
        self.results_history = dict()
        self.dir_checkpoint = dir_checkpoint
        self.load_model = load_model

    def train_model(self):

        if self.is_train:
            try:
                if not self.load_model:
                    self.load_model, self.dir_checkpoint = utils.find_last_version_model(
                        self.dir_checkpoint, self.type_load_model)
            except:
                self.load_model = False

            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f'Using device {device}')

            if self.load_model:
                self.net.load_state_dict(torch.load(
                    self.load_model, map_location=device)['state_dict'])
                logging.info(f'Model loaded from {self.load_model}')
            else:
                logging.info(f'Train new model')
            self.net.to(device=device)
            try:
                self._train()
            except KeyboardInterrupt:
                Path(self.dir_checkpoint).mkdir(parents=True, exist_ok=True)
                self.checkpoint_dict['train_loss'] = self.net_train_loss
                self.checkpoint_dict['valid_loss'] = self.net_valid_loss
                self.checkpoint_dict['valid_accuracy'] = self.net_valid_accuracy
                self.checkpoint_dict['lr'] = self.learning_rates_history[-1]
                self.checkpoint_dict['global_steps'] = self.steps_history[-1]
                self.checkpoint_dict['epochs'] = self.current_epoch
                self.checkpoint_dict['state_dict'] = self.net.state_dict()
                torch.save(self.checkpoint_dict, str(
                    self.dir_checkpoint)+'/INTERRUPTED.pth')
                logging.info('Saved interrupt')
                raise
        else:
            print("No need for training")

    def display_results(self):
        print("Under development")
        if self.dir_checkpoint is None:
            print("There is no dir_checkpoint")
            return
        vn = self.dir_checkpoint.split('/')[-1]
        # dir_results = Path(f'results/{vn}').mkdir(parents=True, exist_ok=True)

        # display all results plotly
        # save them in results
        results_history = self.get_results()
        train_loss = results_history['loss']['train']
        val_loss = results_history['loss']['val']

        valid_score = results_history['accuracy']['val']
        valid_score = [s.item()for s in valid_score]

        learning_rates_history = results_history['learning_rates']

        self.epoch = results_history['epoch']
        steps = results_history['steps']

        # ################################### review
        # fig = make_subplots(
        #     rows=2, cols=2,
        #     specs=[[{"colspan": 2}, None],
        #            [{}, {}]],
        #     subplot_titles=("Train loss", "valid_score", "learning_rates_history"))

        # fig.add_trace(go.Scatter(name='Train', x=list(range(1, global_steps + 1)), y=train_loss),
        #               row=1, col=1)

        # fig.add_trace(go.Scatter(name='valid', x=list(range(1, len(valid_score) + 1)), y=valid_score),
        #               row=2, col=1)
        # fig.add_trace(go.Scatter(name='lr', x=list(range(1, len(learning_rates_history) + 1)), y=learning_rates_history),
        #               row=2, col=2)
        # fig.update_layout(height=400, width=600, showlegend=False,
        #                   title_text=f"Results plane {self.plane}")
        # fig.show()
        pass

    def get_results(self):

        self.results_history['loss'] = {"train": self.train_loss_history,
                                        "val": self.valid_loss_history}
        self.results_history['accuracy'] = {"val": self.valid_accuracy_history}
        self.results_history['learning_rates'] = self.learning_rates_history
        self.results_history['epoch'] = list(range(1, self.current_epoch + 1))
        self.results_history['steps'] = self.steps_history

        return self.results_history

    def _train(self):

        # find what is the existed device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(device)

        last_num = utils.find_last_sub_folders_files('checkpoints')
        self.dir_checkpoint = f'./checkpoints/v{last_num}'
        Path(self.dir_checkpoint).mkdir(parents=True, exist_ok=True)
        best_checkpoint_file = str(
            self.dir_checkpoint) + '/best_checkpoint.pth'
        self.early_stopping = EarlyStopping(
            self.delay, self.checkpoint_dict, checkpoint_save=best_checkpoint_file)

        # (Initialize logging)
        if not self.debug:
            experiment = wandb.init(
            project=f'Clothing Classifier', entity='muhammed-elyamani')
            experiment.config.update(dict(epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate,
              save_checkpoint=self.save_checkpoint, img_size=self.img_size))

        trainloader = self.data_loaders['train']
        validloader = self.data_loaders['val']
        n_train = len(self.image_datasets['train'])
        n_val = len(self.image_datasets['val'])
        classes_names = self.image_datasets['train'].class_names
        logging.info(f'''Starting training:
                Epochs:          {self.epochs}
                Batch size:      {self.batch_size}
                Learning rate:   {self.learning_rate}
                Training size:   {n_train}
                valid size: {n_val}
                Checkpoints:     {self.save_checkpoint}
                Device:          {device.type}
                Images size:  {self.img_size}
            ''')

        if self.print_every is None:
            # every 10 % epoch
            self.print_every = int(0.1 * len(trainloader))
        # intial helping variables
        global_steps = 0
        # loop over epochs
        for epoch in range(1, self.epochs+1):
            self.current_epoch = epoch
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{self.epochs}', unit='img') as pbar:
                # intial helping variables
                running_loss = 0
                # loop over batchs of trainloader
                for images, labels in trainloader:
                    self.net.train()
                    global_steps += 1
                    labels, images = labels.to(device), images.to(device)

                    # clear gradient
                    self.optimizer.zero_grad()
                    # forward pass
                    log_ps = self.net(images)
                    # get loss
                    loss = self.criterion(log_ps, labels)
                    # backward pass
                    loss.backward()
                    # update weights by making step for optimizer
                    self.optimizer.step()

                    running_loss += loss.item()
                    pbar.update(images.shape[0])

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # valid condition every print_every
                    if global_steps % self.print_every == 0:
                        train_loss = running_loss / self.print_every
                        valid_loss, valid_accuracy = evaluate(
                            self.net, self.criterion, validloader)

                        self.train_loss_history.append(train_loss)
                        self.valid_loss_history.append(valid_loss)
                        self.learning_rates_history.append(
                            self.optimizer.param_groups[0]['lr'])
                        self.valid_accuracy_history.append(valid_accuracy)
                        self.steps_history.append(global_steps)
                        if not self.debug:
                            # histograms = {}
                            # for tag, value in self.net.named_parameters():
                            #     tag = tag.replace('/', '.')
                            #     histograms['Weights/' +
                            #                tag] = wandb.Histogram(value.data.cpu())
                            #     histograms['Gradients/' +
                            #                tag] = wandb.Histogram(value.grad.data.cpu())
                            experiment.log({
                                'train loss': train_loss,
                                'valid loss': valid_loss,
                                'learning rate': self.optimizer.param_groups[0]['lr'],
                                'valid accuracy': valid_accuracy,
                                'images': wandb.Image(images[0].cpu()),
                                'labels': {
                                    'true': classes_names[labels[0].cpu()],
                                    'pred': classes_names[get_predictions(log_ps).squeeze()[0].cpu()]
                                },
                                'step': global_steps,
                                'epoch': epoch,
                                # **histograms
                            })
                            wandb.watch(self.net)

                        running_loss = 0
                        # logging.info_results
                        logging.info(
                            f"\n{epoch}/{self.epochs} .. train_loss: {(train_loss) :0.3f}.. valid_loss: {(valid_loss) :0.3f} .. valid_accuracy: {(valid_accuracy * 100) :0.3f}%")

                if self.early_stopping.track(valid_accuracy, self.net, train_loss, valid_loss, self.learning_rates_history, global_steps, self.current_epoch):
                    logging.info("Early stopping")
                    logging.info("Having the best model")
                    self.net = self.early_stopping.get_the_best_model()
                    train_loss, valid_loss, valid_accuracy = self.early_stopping.measurements()
                    logging.info(
                        f"\n.. train_loss: {(train_loss) :0.3f}.. valid_loss: {(valid_loss) :0.3f} .. valid_accuracy: {(valid_accuracy * 100) :0.3f}%")
                    break

            if self.save_checkpoint:
                save_file_checkpoint = ''
                if not self.load_model:

                    save_file_checkpoint = str(self.dir_checkpoint) + \
                        '/checkpoint_epoch{}.pth'.format(epoch)

                else:
                    parent_load_model = '/'.join(
                        self.load_model.split('/')[:-1])
                    last_num = utils.find_last_sub_folders_files(
                        parent_load_model)
                    save_file_checkpoint = f'{parent_load_model}/checkpoint_epoch{last_num}.pth'

                self.checkpoint_dict['train_loss'] = train_loss
                self.checkpoint_dict['valid_loss'] = valid_loss
                self.checkpoint_dict['valid_accuracy'] = valid_accuracy
                self.checkpoint_dict['lr'] = self.learning_rates_history[-1]
                self.checkpoint_dict['global_steps'] = global_steps
                self.checkpoint_dict['epochs'] = self.current_epoch
                self.checkpoint_dict['state_dict'] = self.net.state_dict()

                torch.save(self.checkpoint_dict, save_file_checkpoint)
                logging.info(f'Checkpoint {epoch} saved!')
        # plot train_loss and valid_loss
        plt.plot(self.train_loss_history,  label="Train loss")
        plt.plot(self.valid_loss_history,  label="Valid loss")
        plt.legend()
        plt.show()

        logging.info("Having the best model")
        self.net = self.early_stopping.get_the_best_model()
        train_loss, valid_loss, valid_accuracy = self.early_stopping.measurements()
        logging.info(
            f"\n.. train_loss: {(train_loss) :0.3f}.. valid_loss: {(valid_loss) :0.3f} .. valid_accuracy: {(valid_accuracy * 100) :0.3f}%")

        self.net_train_loss = train_loss
        self.net_valid_loss = valid_loss
        self.net_valid_accuracy = valid_accuracy
