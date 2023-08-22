import torch
from torch import autograd
import numpy as np
import pytorch_lightning as pl
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits, softmax
from torchmetrics import BLEUScore

class T5ModelForPreTraining(pl.LightningModule):
    def __init__(self, config, tokenizer) -> None:
        super().__init__()        
        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=config.num_virtual_tokens)
        lm = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR/ke-t5-large")
        self.model = get_peft_model(lm, peft_config)
        self.model.print_trainable_parameters()

        self.config = config
        self.tokenizer = tokenizer
        self.save_hyperparameters()
        

    def forward(self, batch):
        # (batch_size, sequence_length, config.vocab_size)
        batch = {k: v for k, v in batch.items()}
        return self.model(**batch).logits
    
    
    def training_step(self, batch, batch_idx):
        autograd.set_detect_anomaly(True)
        batch = {k: v for k, v in batch.items()}
        result = self.model(**batch)
        
        loss = result.loss
        ppl = torch.exp(loss)

        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("train_ppl", ppl, on_step=True, prog_bar=True, logger=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        batch = {k: v for k, v in batch.items()}
        result = self.model(**batch)
        
        loss = result.loss
        ppl = torch.exp(loss)
        
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_ppl", ppl, sync_dist=True)


    def test_step(self, batch, batch_idx):
        batch = {k: v for k, v in batch.items()}
        result = self.model(**batch)
        
        loss = result.loss
        ppl = torch.exp(loss)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_ppl", ppl, sync_dist=True)

        
    
    def generate(self, input_ids):
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        
        return self.model.generate(input_ids=input_ids, max_length=512)
        
        
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=2e-5, eps=1e-4)
        # optim = FusedAdam(self.parameters(), lr=2e-5, eps=1e-8)
        
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {"optimizer": optim, "lr_scheduler": scheduler},