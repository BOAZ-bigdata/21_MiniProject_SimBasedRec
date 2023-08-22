import wandb
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from model import AlpacaModelForPreTraining
from data import DataPipeline


def train(config):
    devices = None
    accelerator = None
    if config.device == -1:
        accelerator = "cpu"
    else:
        accelerator = "gpu"
        
        temp = config.device.split(",")
        devices = [int(x) for x in temp]
    
    tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-llama-1-7b")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("-"*10 + "Tokenizer initialized!" + "-"*10)
    
    model  = AlpacaModelForPreTraining(config=config, tokenizer=tokenizer)
    print("-"*10 + "Model initialized!" + "-"*10)

    data_pipeline = DataPipeline(config=config, tokenizer=tokenizer)
    train_dataloader, valid_dataloader, test_dataloader = data_pipeline()
    
    wandb_logger = WandbLogger(project=config.wandb_project, name=f"Koalpaca-Prompt-Tuning{config.batch_size}")
    wandb_logger.experiment.config["batch_size"] = config.batch_size
    print("-"*10 + "Wandb Setting Complete!" + "-"*10)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath='./checkpoint',
                                          filename= f"KoAlpt-batch_size{config.batch_size}"+'-{val_loss:.2f}',
                                          save_top_k=1,
                                          save_last=False,
                                          verbose=True,
                                          mode="min")
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        mode='min',
        check_finite=False,
        patience=3,
    )
    
    trainer = pl.Trainer(
                         accelerator=accelerator,
                         devices=devices,
                         precision=config.precision,
                        #  strategy=config.strategy,
                         enable_progress_bar=True,
                         callbacks=[checkpoint_callback, early_stopping],
                         max_epochs=config.max_epochs,
                         num_sanity_val_steps=config.num_sanity_val_steps,
                         logger=wandb_logger
                         )
    
    print("-"*10 + "Train Start!" + "-"*10)
    
    trainer.fit(model, train_dataloader, valid_dataloader)
    print("-"*10 + "Train Finished!" + "-"*10)

    trainer.test(model, test_dataloader)
    
    wandb.finish()