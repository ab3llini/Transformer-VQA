import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


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
                 num_workers=2,
                 tensorboard=None,
                 checkpoint_path=None,
                 logging_fp=None,
                 logging_fn=None,
                 logging_interval=10,
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
        :param logging_fp: A file descriptor to log training & testing info
        :param logging_fn: A function that defines what to write. Called with model_output, current_batch and desc.
        :param logging_interval: After how many batched logging is performed
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
        self.logging_fp = logging_fp
        self.logging_fn = logging_fn
        self.logging_interval = logging_interval

        self.tr_loader = DataLoader(dataset=self.tr_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
                                    num_workers=self.num_workers)
        self.ts_loader = DataLoader(dataset=self.ts_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
                                    num_workers=self.num_workers)
        # Move model to device
        self.model.to(device)

        # Null gradients
        self.model.zero_grad()

        # Init step
        self.global_step = 0

    def __test(self, epoch):

        print('\nEvaluating model..')

        total_loss = 0
        total_batches = 0

        with torch.no_grad():
            for it, batch in enumerate(tqdm(self.ts_loader)):

                # Unpack batch if need to
                if self.batch_extractor is not None:
                    batch = self.batch_extractor(batch)

                # Move tensors to device
                for idx, o in enumerate(batch):
                    batch[idx] = o.to(self.device)

                # Computing model output
                out = self.model(*batch)

                # Compute the loss
                total_loss += self.loss(out, batch).item()
                total_batches += 1

            loss = total_loss / total_batches

            # Visual debug of progress
            if self.tensorboard is not None:
                self.tensorboard.add_scalar('Loss/test', loss, epoch)
            if self.logging_fp is not None:
                self.logging_fp.write(self.logging_fn(out, batch, 'Test loss = {}'.format(loss)))

    def train(self):

        self.global_step = 0

        print('\nTraining model..')

        for epoch in range(self.epochs):

            iterations = tqdm(self.tr_loader)

            for it, batch in enumerate(iterations):

                # Unpack batch if need to
                if self.batch_extractor is not None:
                    batch = self.batch_extractor(batch)

                # Move tensors to device
                for idx, o in enumerate(batch):
                    batch[idx] = o.to(self.device)

                # Computing model output
                out = self.model(*batch)

                # Compute the loss
                loss_out = self.loss(out, batch)

                loss_out.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.logging_fp is not None:
                    if it % self.logging_interval == 0:
                        self.logging_fp.write(self.logging_fn(out, batch, 'Train loss = {}'.format(loss_out.item())))

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
            self.__test(epoch)
