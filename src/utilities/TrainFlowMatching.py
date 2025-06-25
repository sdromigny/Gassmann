import os
import sys
current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, '../../..'))
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
from utilities.FlowMatchingEstimator import FlowMatchingEstimator
from sbi.neural_nets.embedding_nets import FCEmbedding

import os
import torch
import wandb
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from plotting.Tomography2d import tomo_plot
from plotting.MeshPlot import mesh_plot

class TrainFlowMatchTomo(LightningModule):
    def __init__(self, config, wandb_logger):
        super().__init__()
        self.config = config
        self.wandb_logger = wandb_logger
        self.lr = config.lr
        self.warmup = config.warmup
        self.batch_size = config.batch_size

        self.dim = config.dim
        self.data_dim=config.data_dim
        self.lbound=config.lbound
        self.ubound=config.ubound
        self.sigma = torch.tensor([config.sigma])
        self.N = config.N
        self.d_obs = config.d_obs
        self.batch_plot = config.batch_plot
        self.configfile=config.configfile

        
        self.save_fig = os.path.join(config.path, "fmpe_test.png")

        self.h= config.hidden_units
        self.vector_field = SimpleVectorFieldNet(input_dim=self.dim, condition_dim=self.data_dim, time_encoding_dim=6, hidden_dim=self.h)

        self.embedding = nn.Identity() if config.embedding == 0 else nn.Linear(self.dim, config.embedding)
        self.estimator = FlowMatchingEstimator(
                        net=self.vector_field,
                        input_shape=torch.Size([self.dim]),
                        condition_shape=torch.Size([self.data_dim]),
                        embedding_net=self.embedding, noise_scale=config.noise_scale
                    )

        self.best_loss = float('inf')

        self.best_model_path = os.path.join(config.path, "saved_nn_weights/fm_model1.pth")
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
    
    def plot(self,fontsize):

        # Create the pairplot (this is your existing pairplot function)
        # Generate the observed data pdf
        d_obs =  self.d_obs
        d_obs_tensor = torch.from_numpy(d_obs).float().to(self.device)
        data_batched = d_obs_tensor.unsqueeze(0).repeat(self.N, 1)
        d_pdf = data_batched + self.sigma * torch.randn(self.N, self.data_dim, device=self.device)  
        # Efficient batched sampling
        samples = []
        with torch.no_grad():
            for i in range(0, self.N, self.batch_plot):
                batch_d_test = d_pdf[i : i + self.batch_plot].float()  # Ensure float32
                flow = self.estimator.flow(batch_d_test)
                batch_samples = flow.sample(torch.Size([batch_d_test.shape[0]])).view(-1, self.dim)
                samples.append(batch_samples)

        samples = torch.cat(samples, dim=0)

        # Convert to NumPy before plotting
        # Plot
        samples_np = samples.cpu().numpy()

        # Compute stats
        samples_mean = samples_np.mean(axis=0)
        samples_std  = samples_np.std(axis=0)
        one_sample   = samples_np[-1]             # (100,)

        fig = tomo_plot(samples_mean,samples_std,one_sample,fontsize,self.save_fig)
        # Log the figure to WandB
        self.logger.experiment.log({"2D Tomography": wandb.Image(fig)})
        
        return fig
    
    def training_step(self, batch, batch_idx):
        x1, x0_true = batch  # unpack the tuple
        x1 = x1.to(self.device)
        x0_true = x0_true.to(self.device)
        loss = self.estimator.loss(x1, x0_true).mean()

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

    # def training_step(self, batch, batch_idx):
    #     # Sample from prior
    #     x1 = batch
    #     # Generate synthetic observations
    #     x0 = run_tomo2d(self.configfile, x1.cpu().numpy().astype(np.float64)).to(self.device)
        
    #     loss = self.estimator.loss(x1, x0).mean()
    #     self.loss_hist.append(loss.item())

    #     # Log the loss for monitoring
    #     self.log(
    #         "train_loss",
    #         loss,
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=False,
    #         logger=True,
    #         batch_size=self.batch_size,
    #     )

    #     return loss

    # def train_dataloader(self):
    #     x1_dist = torch.distributions.Uniform(self.lbound, self.ubound)
    #     x1 = x1_dist.sample((self.batch_size, self.dim)).to(self.device)
    #     dataset = x1
    #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


    # def val_dataloader(self):
    #     x1_dist = torch.distributions.Uniform(self.lbound, self.ubound)
    #     x1 = x1_dist.sample((self.batch_size, self.dim)).to(self.device)
    #     dataset = x1
    #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    # def _generate_new_train_dataloader(self):
    #     x1_dist = torch.distributions.Uniform(self.lbound, self.ubound)
    #     x1 = x1_dist.sample((self.batch_size, self.dim)).to(self.device)
    #     dataset = x1
    #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


    # def _generate_new_val_dataloader(self):
    #     x1_dist = torch.distributions.Uniform(self.lbound, self.ubound)
    #     x1 = x1_dist.sample((self.batch_size, self.dim)).to(self.device)
    #     dataset = x1
    #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    
    # def on_train_epoch_start(self):
    #     # Resample the training data for a new epoch
    #     self.train_loader = self._generate_new_train_dataloader()


    # def on_validation_epoch_start(self):
    #     # Resample the validation data for a new epoch
    #     self.val_loader= self._generate_new_val_dataloader()

    


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
        x1, x0_true = batch  # unpack the tuple
        x1 = x1.to(self.device)
        x0_true = x0_true.to(self.device)
        loss = self.estimator.loss(x1, x0_true).mean()

                # Log the loss for monitoring
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )

        return loss

    def on_validation_epoch_end(self):
        pass
    






class TrainFlowMatchPetro(LightningModule):
    def __init__(self, config, wandb_logger):
        super().__init__()
        self.config = config
        self.wandb_logger = wandb_logger
        self.lr = config.lr
        self.warmup = config.warmup
        self.batch_size = config.batch_size
        self.grid=config.grid
        self.dim = config.dim
        self.data_dim=config.data_dim
        self.lbound=config.lbound
        self.ubound=config.ubound
        self.sigma = torch.tensor([config.sigma])
        self.N = config.N
        self.d_obs = config.d_obs
        self.batch_plot = config.batch_plot

        
        self.save_fig = os.path.join(config.path, "fmpe_test.png")

        self.h= config.hidden_units
        # Suppose FCEmbedding maps from data_dim → dim.
        self.embedding = FCEmbedding(
            input_dim=self.data_dim,
            output_dim=self.dim,    # ⟵ this is the size of the embedded vector
            num_layers=5,
            num_hiddens=self.h,
        )

        # … then pass condition_dim = embedding's output_dim = dim, not data_dim …
        self.vector_field = SimpleVectorFieldNet(
            input_dim=self.dim,             # θ has length = dim
            condition_dim=self.dim,         # embedded x has length = dim
            time_encoding_dim=6,
            hidden_dim=self.h,
        )
        self.estimator = FlowMatchingEstimator(
                        net=self.vector_field,
                        input_shape=torch.Size([self.dim]),
                        condition_shape=torch.Size([self.data_dim]),
                        embedding_net=self.embedding, noise_scale=config.noise_scale
                    )

        self.best_loss = float('inf')

        self.best_model_path = os.path.join(config.path, "saved_nn_weights/fm_model1.pth")
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
    
    def plot(self,fontsize):

        # Create the pairplot (this is your existing pairplot function)
        # Generate the observed data pdf
        print("starting integrating")
        d_obs =  self.d_obs
        d_obs_tensor = torch.from_numpy(d_obs).float().to(self.device)
        data_batched = d_obs_tensor.unsqueeze(0).repeat(self.N, 1)
        d_pdf = data_batched + self.sigma * torch.randn(self.N, self.data_dim, device=self.device)  
        # Efficient batched sampling
        samples = []
        with torch.no_grad():
            for i in range(0, self.N, self.batch_plot):
                batch_d_test = d_pdf[i : i + self.batch_plot].float()  # Ensure float32
                flow = self.estimator.flow(batch_d_test)
                batch_samples = flow.sample(torch.Size([batch_d_test.shape[0]])).view(-1, self.dim)
                samples.append(batch_samples)

        samples = torch.cat(samples, dim=0)

        # Convert to NumPy before plotting
        # Plot
        samples_np = samples.cpu().numpy()

        # Compute stats
        samples_mean = samples_np.mean(axis=0)
        samples_std  = samples_np.std(axis=0)
        one_sample   = samples_np[-1]             # (100,)

        fig = mesh_plot(self.grid,samples_mean,samples_std,one_sample,fontsize,self.save_fig)
        # Log the figure to WandB
        self.logger.experiment.log({"2D Tomography": wandb.Image(fig)})
        
        return fig
    
    def training_step(self, batch, batch_idx):
        x1, x0_true = batch  # unpack the tuple
        x1 = x1.to(self.device)
        x0_true = x0_true.to(self.device)
        loss = self.estimator.loss(x1, x0_true).mean()

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

    # def training_step(self, batch, batch_idx):
    #     # Sample from prior
    #     x1 = batch
    #     # Generate synthetic observations
    #     x0 = run_tomo2d(self.configfile, x1.cpu().numpy().astype(np.float64)).to(self.device)
        
    #     loss = self.estimator.loss(x1, x0).mean()
    #     self.loss_hist.append(loss.item())

    #     # Log the loss for monitoring
    #     self.log(
    #         "train_loss",
    #         loss,
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=False,
    #         logger=True,
    #         batch_size=self.batch_size,
    #     )

    #     return loss

    # def train_dataloader(self):
    #     x1_dist = torch.distributions.Uniform(self.lbound, self.ubound)
    #     x1 = x1_dist.sample((self.batch_size, self.dim)).to(self.device)
    #     dataset = x1
    #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


    # def val_dataloader(self):
    #     x1_dist = torch.distributions.Uniform(self.lbound, self.ubound)
    #     x1 = x1_dist.sample((self.batch_size, self.dim)).to(self.device)
    #     dataset = x1
    #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    # def _generate_new_train_dataloader(self):
    #     x1_dist = torch.distributions.Uniform(self.lbound, self.ubound)
    #     x1 = x1_dist.sample((self.batch_size, self.dim)).to(self.device)
    #     dataset = x1
    #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


    # def _generate_new_val_dataloader(self):
    #     x1_dist = torch.distributions.Uniform(self.lbound, self.ubound)
    #     x1 = x1_dist.sample((self.batch_size, self.dim)).to(self.device)
    #     dataset = x1
    #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    
    # def on_train_epoch_start(self):
    #     # Resample the training data for a new epoch
    #     self.train_loader = self._generate_new_train_dataloader()


    # def on_validation_epoch_start(self):
    #     # Resample the validation data for a new epoch
    #     self.val_loader= self._generate_new_val_dataloader()

    


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
    
    
    def training_step(self, batch, batch_idx):
        x1, x0_true = batch  # unpack the tuple
        x1 = x1.to(self.device)
        x0_true = x0_true.to(self.device)
        loss = self.estimator.loss(x1, x0_true).mean()

                # Log the loss for monitoring
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )

        return loss

    def on_validation_epoch_end(self):
        pass
    


