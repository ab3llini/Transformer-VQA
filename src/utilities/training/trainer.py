import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from PIL import Image


class Trainer:
    def __init__(self,
                 model,
                 tr_dataset,
                 ts_dataset,
                 optimizer,
                 loss,
                 lr,
                 batch_size=64,
                 batch_extractor=None,
                 epochs=10,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 num_workers=4,
                 shuffle=True,
                 tensorboard=None,
                 checkpoint_path=None,
                 callback_fn=None,
                 callback_interval=10,
                 ):
        """
        This class makes training a model easier.
        Use it to quickly run training & testing on any model
        :param model: Actual model instance.
        :param tr_dataset: A Dataset instance with training data
        :param ts_dataset: A Dataset instance with testing data
        :param optimizer: An instance of thew optimizer to use already set up
        :param loss: a loss function. Will be called with model_out, current_batch
        :param lr: Learning rate in use
        :param batch_size: Size of each batch
        :param batch_extractor: Function to manipulate every batch before passing it to the model. Optional
        :param epochs: Number of epochs to train for
        :param device: Device 'cpu' or 'cuda' (Default)
        :param num_workers: Number of processes to load data from the loaders
        :param tensorboard: A SummaryWriter instance. Optional
        :param checkpoint_path: A valid path to the checkpoints folder
        :param callback_fn: Call back function:
        Passed args: output, batch, iteration, epoch, task
        :param callback_interval: Callback interval
        """

        self.model = model
        self.tr_dataset = tr_dataset
        self.ts_dataset = ts_dataset
        self.optimizer = optimizer
        self.loss = loss
        self.lr = lr
        self.batch_extractor = batch_extractor
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.num_workers = num_workers
        self.tensorboard = tensorboard
        self.checkpoint_path = checkpoint_path
        self.callback_fn = callback_fn
        self.callback_interval = callback_interval
        self.shuffle = shuffle

        # Move model to device
        self.model.to(device)

        # Null gradients
        self.model.zero_grad()

        # Init step
        self.global_step = 0

    def __test(self, epoch):

        ts_loader = DataLoader(dataset=self.ts_dataset, shuffle=self.shuffle, batch_size=self.batch_size, pin_memory=True,
                               num_workers=self.num_workers)

        print('\nEvaluating model..\n')

        total_loss = 0
        total_batches = 0

        with torch.no_grad():
            for it, batch in enumerate(tqdm(ts_loader)):

                # Unpack batch if need to
                if self.batch_extractor is not None:
                    extracted_batch = self.batch_extractor(batch)
                else:
                    extracted_batch = batch

                # Move tensors to device
                for idx, o in enumerate(extracted_batch):
                    extracted_batch[idx] = o.to(self.device)

                # Computing model output
                out = self.model(*extracted_batch)

                # Compute the loss
                total_loss += self.loss(out, extracted_batch).item()
                total_batches += 1

                if self.callback_fn is not None:
                    if it > 0 and it % self.callback_interval == 0:
                        self.callback_fn(out, batch, it, epoch, 'test')

            loss = total_loss / total_batches

            # Visual debug of progress
            if self.tensorboard is not None:
                self.tensorboard.add_scalar('Loss/test', loss, epoch)

        print('\nEvaluation completed.\n')

    def train(self):

        tr_loader = DataLoader(dataset=self.tr_dataset, shuffle=self.shuffle, batch_size=self.batch_size, pin_memory=True,
                                    num_workers=self.num_workers)

        self.global_step = 0

        print('\nTraining model..\n')

        for epoch in range(self.epochs):

            iterations = tqdm(tr_loader)

            for it, batch in enumerate(iterations):

                # Unpack batch if need to
                if self.batch_extractor is not None:
                    extracted_batch = self.batch_extractor(batch)
                else:
                    extracted_batch = batch

                # Move tensors to device
                for idx, o in enumerate(extracted_batch):
                    extracted_batch[idx] = o.to(self.device)

                # Computing model output
                out = self.model(*extracted_batch)

                # Compute the loss
                loss_out = self.loss(out, extracted_batch)

                loss_out.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.callback_fn is not None:
                    if it > 0 and it % self.callback_interval == 0:
                        self.callback_fn(out, batch, it, epoch, 'train')

                # Visual debug of progress
                iterations.set_description('Epoch : {}/{} - Loss: {}'.format(epoch + 1, self.epochs, loss_out.item()))
                if self.tensorboard is not None:
                    self.tensorboard.add_scalar('Loss/train', loss_out.item(), self.global_step)

                self.global_step += 1

            torch.save(self.model.state_dict(),
                       os.path.join(self.checkpoint_path, 'B_{}_LR_{}_CHKP_EPOCH_{}.pth'.format(self.batch_size,
                                                                                                self.lr,
                                                                                                epoch)))

            # Evaluate test loss every epoch
            if self.ts_dataset is not None:
                self.__test(epoch)
