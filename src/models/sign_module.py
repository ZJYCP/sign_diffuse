from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from src.models.components.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)
from torchmetrics import MinMetric, MeanMetric

from src.utils.plot_videos import plot_video
from src.utils.utils import saveSkels
import pickle
import os

class SignLitModule(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        *args, **kargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore='net')

        sampler = 'uniform'
        beta_scheduler = 'linear'
        betas = get_named_beta_schedule(beta_scheduler, self.hparams.diffusion_steps)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )
        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
        self.sampler_name = sampler

        self.net = net

        self.criterion = torch.nn.MSELoss(reduction='none')

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, batch):

        text, gloss, motions, m_lens = batch
        caption = gloss

        x_start = motions
        B, T = x_start.shape[:2]

        cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)

        t, _ = self.sampler.sample(B, x_start.device)
        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_start,
            t=t,
            model_kwargs={"text": caption, "length": cur_len}
        )

        return output, T, cur_len


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_loss_best.reset()

    def step(self, batch: Any):
        output, T, cur_len = self.forward(batch)
        real_noise = output['target']
        fake_noise = output['pred']        
        try:
            self.src_mask = self.net.module.generate_src_mask(T, cur_len).to(self.device)
        except:
            self.src_mask = self.net.generate_src_mask(T, cur_len).to(self.device)

        loss_mot_rec = self.criterion(fake_noise, real_noise).mean(dim=-1)
        loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()
        
        return loss_mot_rec, fake_noise, real_noise

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        text, gloss, motions, m_lens = batch
        caption = gloss
        xf_proj, xf_out = self.net.encode_text(caption, self.device)
        B = len(caption)
        # T = min(m_lens.max(), self.net.num_frames)
        T = self.net.num_frames
        output = self.diffusion.p_sample_loop(
            self.net,
            (B, T, self.net.input_feats),
            clip_denoised=False,
            progress=False,
            model_kwargs={
                'xf_proj': xf_proj,
                'xf_out': xf_out,
                'length': m_lens
            })

        cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)            
        try:
            self.src_mask = self.net.module.generate_src_mask(T, cur_len).to(self.device)
        except:
            self.src_mask = self.net.generate_src_mask(T, cur_len).to(self.device)

        loss_mot_rec = self.criterion(output, motions).mean(dim=-1)
        loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()

        # loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss_mot_rec)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss_mot_rec, "preds": output, "targets": motions}

    def validation_epoch_end(self, outputs: List[Any]):
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        # if batch_idx > 0:
        #     return
        text, gloss, motions, m_lens = batch
        caption = gloss
        xf_proj, xf_out = self.net.encode_text(caption, self.device)
        B = len(caption)
        # T = min(m_lens.max(), self.net.num_frames)
        T = self.net.num_frames
        output = self.diffusion.p_sample_loop(
            self.net,
            (B, T, self.net.input_feats),
            clip_denoised=False,
            progress=False,
            model_kwargs={
                'xf_proj': xf_proj,
                'xf_out': xf_out,
                'length': m_lens
            })
        if batch_idx < 0:
            for i in range(output.shape[0]):
                pred = output[i].detach().cpu().numpy()[:m_lens[i]]

                plot_video(joints=pred,
                                    file_path=self.hparams['save_path'],
                                    video_name="train_"+str(i),
                                    references=motions[i].detach().cpu().numpy(),
                                    skip_frames=1,
                                    sequence_ID="4")

        return output,gloss,text,m_lens

    def test_epoch_end(self, outputs: List[Any]):
        # saveSkels(gloss, text, motion, dataset, save_name):
        #         Data.append({'name':name,'signer':signer,'gloss':gloss,'text':text, 'sign':sign})
        Data = []
        for items in outputs:
            for i in range(len(items[1])):
                m_len = items[3][i].cpu()
                sign_motion = items[0][i,:m_len].detach().cpu()
                step = (torch.arange(1, m_len+1, step=1) / m_len).unsqueeze(-1)
                sign_motion = torch.cat((sign_motion, step), dim=1).numpy()

                Data.append({'name':'diffuse','signer':'diffusionn','gloss':items[1][i], 'text': items[2][i], 'sign':sign_motion})
        pkl_f=open(os.path.join(self.hparams['save_path'],'sign_best.dev'),'wb')
        pickle.dump(Data,pkl_f)
        pkl_f.close()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        
        if batch_idx > 0:
            return
        text, gloss, motions, m_lens = batch
        caption = gloss
        xf_proj, xf_out = self.net.encode_text(caption, self.device)
        B = len(caption)
        # T = min(m_lens.max(), self.net.num_frames)
        T = self.net.num_frames
        output = self.diffusion.p_sample_loop(
            self.net,
            (B, T, self.net.input_feats),
            clip_denoised=False,
            progress=False,
            model_kwargs={
                'xf_proj': xf_proj,
                'xf_out': xf_out,
                'length': m_lens
            })
        for i in range(output.shape[0]):
            pred = output[i].detach().cpu().numpy()[:m_lens[i]]

            plot_video(joints=pred,
                                file_path=self.hparams['save_path'],
                                video_name="train_"+str(i),
                                references=motions[i].detach().cpu().numpy(),
                                skip_frames=1,
                                sequence_ID="4")

        return output        

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
