import os
import sys
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)

import torch
import torch.nn as nn
import numpy as np
import normflows as nf
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from utilities.MLP import SimpleVectorFieldNet
from utilities.PlotHighD import plot_5d_corner
#from sbi.neural_nets.estimators.flowmatching_estimator import 
from utilities.FlowMatchingEstimator import FlowMatchingEstimator


import os
import torch
import wandb
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

class TrainFlowMatch(LightningModule):
    def __init__(self, config, wandb_logger, simulator):
        super().__init__()
        self.config = config
        self.wandb_logger = wandb_logger
        self.lr = config.lr
        self.warmup = config.warmup
        self.batch_size = config.batch_size

        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.sigma = torch.tensor([config.sigma])
        self.N = config.N
        self.d_obs = config.d_obs
        self.batch_plot = config.batch_plot
        
        self.save_fig = "/home/users/scro4690/Documents/GenInv/Gassmann/src/example/full/fm_test.png"


        self.h= config.hidden_units
        self.vector_field = SimpleVectorFieldNet(input_dim=self.input_dim, condition_dim=self.output_dim, time_encoding_dim=6, hidden_dim=self.h)
        self.simulator = simulator  # Use the actual function, not a string
        self.embedding = nn.Identity() if config.embedding == 0 else nn.Linear(self.dim, config.embedding)
        self.estimator = FlowMatchingEstimator(
                        net=self.vector_field,
                        input_shape=torch.Size([self.input_dim]),
                        condition_shape=torch.Size([self.output_dim]),
                        embedding_net=self.embedding, noise_scale=config.noise_scale
                    )

        self.best_loss = float('inf')

        self.loss_hist = []



    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config=None, wandb_logger=None, **kwargs):
        # Use the default Lightning behavior to load the checkpoint
        instance = super().load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **kwargs
        )

        # Manually set additional attributes if needed
        instance.config = config
        instance.wandb_logger = wandb_logger
        return instance
    
    def plot(self,m_true,fontsize=12):
        
        

        # Create the pairplot (this is your existing pairplot function)
        # Generate the observed data pdf
        d_obs =  self.d_obs
        d_pdf = d_obs + self.sigma * torch.randn(self.N, self.output_dim, device=self.device)
        # Efficient batched sampling
        samples = []
        with torch.no_grad():
            for i in range(0, self.N, self.batch_plot):
                batch_d_test = d_pdf[i : i + self.batch_plot]  # Get batch
                flow = self.estimator.flow(batch_d_test)  # Get flow for batch

                batch_samples = flow.sample(torch.Size([batch_d_test.shape[0]]))  # Sample
                
                batch_samples = flow.sample(torch.Size([batch_d_test.shape[0]])).view(-1, self.input_dim)
                samples.append(batch_samples)


        samples = torch.cat(samples, dim=0)

        # Convert to NumPy before plotting
        samples_np = samples.cpu().numpy()

        fig = plot_5d_corner(samples_np, save_path=self.save_fig)
        # Log the figure to WandB
        self.logger.experiment.log({"2D histogram comparison": wandb.Image(fig)})
        
        return fig

    def training_step(self, batch, batch_idx):
        # Sample from prior
        x1 = batch
        # Generate synthetic observations
        x0 = self.simulator(x1)
        
        # Compute loss
        # estimator=FlowMatchingEstimator(
        #                 net=self.vector_field,
        #                 input_shape=torch.Size([self.dim]),
        #                 condition_shape=torch.Size([self.dim]),
        #                 embedding_net=self.embedding,
        #             )
        loss = self.estimator.loss(x1, x0).mean()

        # Log the loss for monitoring
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )

        return loss

    def train_dataloader(self):
        # Sample prior distribution for training
        uni = torch.distributions.Uniform(0.0, 10.0)
        u = uni.sample((self.batch_size, 2))  # shape (B,2)

        # 2) Sample last three dims Normal(means, stds)
        means = torch.tensor([8.5,  0.37, 44.8])
        stds  = torch.tensor([0.3,  0.02, 0.8 ])
        # standard normals
        z = torch.randn(self.batch_size, 3)
        g = z * stds + means                      # shape (B,3)

        # 3) Concatenate to (B,5)
        x = torch.cat([u, g], dim=1)

        # 4) Return DataLoader as before
        return DataLoader(x, batch_size=self.batch_size, shuffle=True)


    def val_dataloader(self):
        # Sample uniformly from [0, 10] for a 2D tensor of shape (batch_size, dim)
        uni = torch.distributions.Uniform(0.0, 10.0)
        u = uni.sample((self.batch_size, 2))  # shape (B,2)

        # 2) Sample last three dims Normal(means, stds)
        means = torch.tensor([8.5,  0.37, 44.8])
        stds  = torch.tensor([0.3,  0.02, 0.8 ])
        # standard normals
        z = torch.randn(self.batch_size, 3)
        g = z * stds + means                      # shape (B,3)

        # 3) Concatenate to (B,5)
        x = torch.cat([u, g], dim=1)

        # 4) Return DataLoader as before
        return DataLoader(x, batch_size=self.batch_size, shuffle=False)


    def _generate_new_train_dataloader(self):
        # 1) Sample first two dims Uniform(0,10)
        uni = torch.distributions.Uniform(0.0, 10.0)
        u = uni.sample((self.batch_size, 2))  # shape (B,2)

        # 2) Sample last three dims Normal(means, stds)
        means = torch.tensor([8.5,  0.37, 44.8])
        stds  = torch.tensor([0.3,  0.02, 0.8 ])
        # standard normals
        z = torch.randn(self.batch_size, 3)
        g = z * stds + means                      # shape (B,3)

        # 3) Concatenate to (B,5)
        x = torch.cat([u, g], dim=1)

        # 4) Return DataLoader as before
        return DataLoader(x, batch_size=self.batch_size, shuffle=True)

    def _generate_new_val_dataloader(self):
        uni = torch.distributions.Uniform(0.0, 10.0)
        u = uni.sample((self.batch_size, 2))  # shape (B,2)

        # 2) Sample last three dims Normal(means, stds)
        means = torch.tensor([8.5,  0.37, 44.8])
        stds  = torch.tensor([0.3,  0.02, 0.8 ])
        # standard normals
        z = torch.randn(self.batch_size, 3)
        g = z * stds + means                      # shape (B,3)

        # 3) Concatenate to (B,5)
        x = torch.cat([u, g], dim=1)

        # 4) Return DataLoader as before
        return DataLoader(x, batch_size=self.batch_size, shuffle=False)

    
    def on_train_epoch_start(self):
        # Resample the training data for a new epoch
        self.train_loader = self._generate_new_train_dataloader()


    def on_validation_epoch_start(self):
        # Resample the validation data for a new epoch
        self.val_loader= self._generate_new_val_dataloader()

    


    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        class scheduler_lambda_function:
            def __init__(self, warm_up):
                self.use_warm_up = True if warm_up > 0 else False
                self.warm_up = warm_up

            def __call__(self, s):
                # Adjust learning rate based on the current step and warm-up period
                if self.use_warm_up:
                    if s < self.warm_up:
                        return 100 * (self.warm_up - s) / self.warm_up + 1
                    else:
                        return 1
                else:
                    return 1

        # Create an Adam optimizer for the Normalizing Flow model parameters
        optimizer = torch.optim.Adam(self.estimator.parameters(), lr=1e-2)
        # Set up a learning rate scheduler that adjusts the learning rate based on the defined warm-up function

        return [optimizer]
    
    
    def validation_step(self, batch, batch_idx):
        x1 = batch
        # Generate synthetic observations
        x0 = self.simulator(x1)
        
        # Compute loss    # Compute loss
        # estimator=FlowMatchingEstimator(
        #                 net=self.vector_field,
        #                 input_shape=torch.Size([self.dim]),
        #                 condition_shape=torch.Size([self.dim]),
        #                 embedding_net=self.embedding,
        #             )
        loss = self.estimator.loss(x1, x0).mean()

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )

    def on_validation_epoch_end(self):
        pass
    

