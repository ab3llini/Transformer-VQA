import wandb
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
from torch import nn


class Trainer:
    def __init__(self,
                 wandb_args,
                 model,
                 tr_dataset,
                 ts_dataset,
                 optimizer,
                 loss,
                 lr,
                 batch_size,
                 epochs,
                 early_stopping,
                 device,
                 num_workers,
                 checkpoint_path,
                 shuffle,
                 log_interval=100
                 ):

        self.model = model
        self.tr_dataset = tr_dataset
        self.ts_dataset = ts_dataset
        self.optimizer = optimizer
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.device = device
        self.num_workers = num_workers
        self.checkpoint_path = checkpoint_path
        self.shuffle = shuffle
        self.log_interval = log_interval

        self.global_step = 0
        self.last_test_loss = None
        self.since_improvement = None

        wandb.init(**wandb_args)

        wandb.config.learning_rate = lr
        wandb.config.epochs = epochs
        wandb.config.batch_size = batch_size
        wandb.config.device = device
        wandb.config.loss = loss
        wandb.config.optimizer = optimizer

    def run(self):

        if torch.cuda.device_count() > 1:
            dp = input(
                'We detected {} GPUs, do you want to turn DataParallel on? [Y/n]: '.format(
                    torch.cuda.device_count()
                )
            )
            if dp == 'Y' or dp == 'y':
                print('Turning on DataParallel')
                self.model = nn.DataParallel(self.model)
            else:
                print('DataParallel disabled')
                gpu = input('Which gpu would you like to use? [0]: ')
                if gpu != '' and int(gpu) < torch.cuda.device_count():
                    self.device = torch.device('cuda:{}'.format(gpu))

        self.model.to(self.device)
        self.model.zero_grad()
        self.__train()

    def __log(self, epoch, it, its, loss, description, delta=0.0, wandb_log=True):
        print(
            '{} | Epoch: {}/{} | Iter: {}/{} | Loss: {}'.format(
                description,
                epoch + 1,
                self.epochs,
                it,
                its,
                loss,
            ) + ' | Delta: {0:.2f}s'.format(delta)
        )
        if wandb_log:
            wandb.log({"{}".format(description): loss}, step=self.global_step)

    def __train(self):

        loader = DataLoader(
            dataset=self.tr_dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers
        )

        print('Training..')

        for epoch in range(self.epochs):
            running_loss = 0
            epoch_loss = 0
            running_timer = time.time()
            epoch_timer = time.time()
            for it, batch in enumerate(loader):
                # Move tensors to device
                batch = list(map(lambda tensor: tensor.to(self.device), batch))

                self.optimizer.zero_grad()

                # Computing model output
                out = self.model(*batch)

                # Compute the loss
                loss = self.loss(out, batch)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()
                self.global_step += 1

                if (it + 1) % self.log_interval == 0:
                    running_loss /= self.log_interval
                    self.__log(
                        epoch,
                        it + 1,
                        len(loader),
                        running_loss,
                        'Running training loss',
                        time.time() - running_timer
                    )
                    running_loss = 0
                    running_timer = time.time()

            # Done with current epoch
            self.__log(
                epoch,
                len(loader),
                len(loader),
                epoch_loss / len(loader),
                'Training loss',
                time.time() - epoch_timer
            )

            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.checkpoint_path,
                    '{}_bs={}_lr={}_e={}.pth'.format(
                        type(self.model).__name__,
                        self.batch_size,
                        self.lr,
                        epoch
                    )
                )
            )

            stop = self.__test(after_epoch=epoch)
            if stop:
                print('Early stropping triggered!')
                break

    def __test(self, after_epoch):
        loader = DataLoader(
            dataset=self.ts_dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers
        )

        print('Testing..')
        test_loss = 0
        running_loss = 0
        running_timer = time.time()
        test_timer = time.time()

        with torch.no_grad():
            for it, batch in enumerate(loader):

                batch = list(map(lambda tensor: tensor.to(self.device), batch))
                out = self.model(*batch)

                # Compute the loss
                loss = self.loss(out, batch)
                running_loss += loss.item()
                test_loss += loss.item()

                if (it + 1) % self.log_interval == 0:
                    self.__log(
                        after_epoch,
                        it + 1,
                        len(loader),
                        running_loss / self.log_interval,
                        'Testing running loss',
                        time.time() - running_timer,
                        wandb_log=False
                    )
                    running_loss = 0
                    running_timer = time.time()

            test_loss /= len(loader)

            self.__log(
                after_epoch,
                len(loader),
                len(loader),
                test_loss,
                'Testing loss',
                time.time() - test_timer
            )

            # Early stopping
            if self.last_test_loss is None:
                self.last_test_loss = test_loss
                self.since_improvement = 0
                return False
            else:
                if test_loss >= self.last_test_loss:
                    if self.since_improvement == self.early_stopping:
                        # No improvements for more than 3 epochs
                        return True
                    else:
                        self.since_improvement += 1
                        return False
                else:
                    self.last_test_loss = test_loss
                    self.since_improvement = 0
