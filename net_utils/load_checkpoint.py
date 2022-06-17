import torch
from torch import optim
from torchvision import models


# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(file, checkpoint: dict, Network):
    print("Loading the model and the hyperparameters")

    print(f'Model loaded from {file}')
    checkpoint = torch.load(file)
    if checkpoint['arch'] == 'vgg16':
        # load pretrained_model
        model = models.vgg16(pretrained=True)
        # freaze parameters
        for param in model.parameters():
            param.requires_grad = False
        # define classifier
        classifier = Network(checkpoint['input_size'], checkpoint['output_size'],
                             checkpoint['hidden_sizes'], checkpoint['dropout_p'])
        # transfer learning
        model.classifier = classifier
        # load the previus model
        model.load_state_dict(checkpoint['state_dict'])

        model.class_to_idx = checkpoint['class_to_idx']

        # define optimizer
        optimizer = optim.Adam(model.classifier.parameters())
        # load the previus optimizer
        optimizer.load_state_dict(checkpoint['optimzier_state_dict'])
        # turn into optimizer.cuda() if it was in cuda
        cond_cuda = True if torch.cuda.is_available() else False
        if cond_cuda:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        print("Info about checkpoint model:")
        try:
            print("Epoch: {} .. train_loss: {:.3f}, valid_loss: {:.3f}, valid_accuracy: {:.3f}".format(
                checkpoint['epochs'], checkpoint['train_loss'], checkpoint['valid_loss'], checkpoint['valid_accuracy']*100))
        except:
            print('there is no metrics to print')
            pass
        return model, optimizer
