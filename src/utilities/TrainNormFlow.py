import os


import torch
import numpy as np
import normflows as nf
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
#from nn_models.Normflows import NormFlow

import os
import torch
import wandb
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
import configparser

class TomoFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, configfile, observed_data, sigma):
        # z: (batch_size, dim) tensor on device
        # Run simulator on CPU:
        z_np = z.detach().cpu().numpy().astype(np.float64)
        synthetic_value, grads = run_tomo2d(configfile, z_np)
        # synthetic_value: CPU tensor shape (batch_size, n_data)
        # grads: CPU tensor shape (batch_size, n_data, dim)
        device = z.device
        dsim = synthetic_value.to(device)  # move to GPU if needed
        gradients = grads.to(device)       # shape (batch_size, n_data, dim)
        # Save gradients for backward
        ctx.save_for_backward(gradients)
        # Note: observed_data and sigma should be torch tensors on same device, used outside
        return dsim  # shape (batch_size, n_data)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: shape (batch_size, n_data), i.e. ∂loss/∂synthetic_value
        (gradients,) = ctx.saved_tensors  # shape (batch_size, n_data, dim)
        # Ensure grad_output is on same device:
        if grad_output.device != gradients.device:
            grad_output = grad_output.to(gradients.device)
        # Combine over data dim:
        # grad_input shape: (batch_size, dim)
        grad_input = torch.einsum('bn,bnd->bd', grad_output, gradients)
        # Return gradient w.r.t. z, and None for other args
        return grad_input, None, None, None


class TomoFunction_lglike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, configfile):
        # Detach and run simulator
        z_np = z.detach().cpu().numpy().astype(np.float64)
        res, grads = run_tomo2d_lglike(configfile, z_np)
        # Now res and grads are CPU torch.Tensors float32.
        # Move to the same device as z:
        device = z.device
        log_p = res.to(device)  # float32 on correct device
        gradients = grads.to(device)  # shape (batch_size, dim)
        # Save gradients for backward
        ctx.save_for_backward(gradients)
        return log_p

    @staticmethod
    def backward(ctx, grad_output):
        (gradients,) = ctx.saved_tensors
        grad_output = grad_output.view(-1, 1)  # Shape (B, 1)
        grad_input = grad_output * gradients   # Shape (B, D)
        return grad_input, None

    


class TrainNormFlow(LightningModule):
    def __init__(self, config, wandb_logger, nfm):
        super().__init__()
        self.automatic_optimization = False  # Important!
        # “config” here is the argparse Namespace, so:
        #   config.configfile == "/path/to/config_simul.ini"
        self.wandb_logger = wandb_logger
        self.lr           = config.lr
        self.warmup       = config.warmup
        self.batch_size   = config.batch_size

        self.dim      = config.dim
        self.data_dim = config.data_dim
        self.lbound   = config.lbound
        self.ubound   = config.ubound
        self.sigma    = torch.tensor([config.sigma], dtype=torch.float32)
        self.N        = config.N
        # d_obs was passed as a NumPy array via argparse default:
        self.d_obs    = config.d_obs
        self.configfile = config.configfile

        self.save_fig = os.path.join(config.path, "nf_test.png")
        self.nfm      = nfm  # your actual NormFlow model

        self.best_loss = float("inf")
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
        
        z, _ = self.nfm.sample(self.N)
        z_np=z.detach().cpu().numpy()
        samples_mean = z_np.mean(axis=0)
        samples_std  = z_np.std(axis=0)
        one_sample   = z_np[-1]             # (100,)


        fig =tomo_plot(samples_mean,samples_std,one_sample,fontsize,self.save_fig)
        # Log the figure to WandB
        self.logger.experiment.log({"2D histogram comparison": wandb.Image(fig)})
        
        return fig


    # def training_step(self, batch, batch_idx):

    #     observed_data = torch.from_numpy(self.d_obs).float().to(self.device)

    #     z, log_q = self.nfm.forward_and_log_det(batch)

    #     print("log q", log_q)

    #     # Run forward model
    #     synthetic_value = TomoFunction.apply(z, self.configfile, observed_data, self.sigma)

    #     # Make sure sigma is also on the correct device
    #     sigma = self.sigma.to(self.device) if isinstance(self.sigma, torch.Tensor) else torch.tensor(self.sigma, device=self.device)

    #     log_p = -0.5 * (((synthetic_value - observed_data) / sigma) ** 2).sum(dim=1)
    #     print("log p", log_p)
    #     loss = -(log_q + log_p).mean()
    #     self.loss_hist.append(loss.item())

    #     # Log the validation loss for monitoring
    #     self.log(
    #         "loss",
    #         loss,
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=False,
    #         logger=True,
    #         batch_size=self.batch_size,
    #     )
    #     return loss

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        observed_data = torch.from_numpy(self.d_obs).float().to(self.device)
        z, log_q = self.nfm.forward_and_log_det(batch)

        # 2) Convert to torch.Tensor and move to device
        log_p = TomoFunction_lglike.apply(z, self.configfile, observed_data, self.sigma)
        loss = -(log_q + log_p).mean()
        self.loss_hist.append(loss.item())
        print("log p",log_p)
        print("log q",log_q)
        optimizer.zero_grad()
        loss.backward()  # You can use this if still doing standard backward
        optimizer.step()
        # Log the validation loss for monitoring
        self.log(
            "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )
        return loss


    def train_dataloader(self):
        x1_dist = torch.distributions.Uniform(self.lbound, self.ubound)
        x1 = x1_dist.sample((self.batch_size, self.dim)).to(self.device)
        dataset = x1
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


    def val_dataloader(self):
        x1_dist = torch.distributions.Uniform(self.lbound, self.ubound)
        x1 = x1_dist.sample((self.batch_size, self.dim)).to(self.device)
        dataset = x1
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _generate_new_train_dataloader(self):
        x1_dist = torch.distributions.Uniform(self.lbound, self.ubound)
        x1 = x1_dist.sample((self.batch_size, self.dim)).to(self.device)
        dataset = x1
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


    def _generate_new_val_dataloader(self):
        x1_dist = torch.distributions.Uniform(self.lbound, self.ubound)
        x1 = x1_dist.sample((self.batch_size, self.dim)).to(self.device)
        dataset = x1
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    
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
        optimizer = torch.optim.Adam(lr=self.lr, params=self.nfm.parameters())
        # Set up a learning rate scheduler that adjusts the learning rate based on the defined warm-up function
        scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda_function(self.warmup)),
                     'interval': 'epoch'}  # Scheduler is called after each training epoch
        return [optimizer], [scheduler]
    
    
    def validation_step(self, batch, batch_idx):        

        observed_data = torch.from_numpy(self.d_obs).float().to(self.device)

        z, log_q = self.nfm.forward_and_log_det(batch)

        # Run forward model
        synthetic_value, gradients = run_tomo2d(self.configfile, z.detach().cpu().numpy().astype(np.float64))
        synthetic_value=synthetic_value.to(self.device)
        gradients=gradients.to(self.device)
        # Make sure sigma is also on the correct device
        sigma = self.sigma.to(self.device) if isinstance(self.sigma, torch.Tensor) else torch.tensor(self.sigma, device=self.device)

        log_p = -0.5 * (((synthetic_value - observed_data) / sigma) ** 2).sum(dim=1)
        loss = -(log_q + log_p).mean()
        self.loss_hist.append(loss.item())
        print("log p",log_p)
        print("log q",log_q)
        

        # Log the validation loss for monitoring
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )

    # def validation_step(self, batch, batch_idx):
    #     z, log_q = self.nfm.forward_and_log_det(batch)
        
    #     # 1) Call the Fortran wrapper → NumPy array
    #     res_np = run_tomo2d_lglike(self.configfile, 
    #                             z.detach().cpu().numpy().astype(np.float64))

    #     # 2) Convert to torch.Tensor and move to device
    #     log_p = torch.from_numpy(res_np).float().to(batch.device)
    #     loss = -(log_q + log_p).mean()
    #     self.loss_hist.append(loss.item())
        

    #     # Log the validation loss for monitoring
    #     self.log(
    #         "val_loss",
    #         loss,
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=False,
    #         logger=True,
    #         batch_size=self.batch_size,
    #     )

    def on_validation_epoch_end(self):
        pass
    


