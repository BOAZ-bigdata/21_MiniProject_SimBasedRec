from typing import *
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
import datasets
from datasets import load_dataset


class DataPipeline:
    def __init__(self, config, tokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer

    def __call__(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dataloader, valid_dataloader, test_dataloader = self.data_pipeline()
        
        return train_dataloader, valid_dataloader, test_dataloader

        
    def load_data(self):
        data_st = load_dataset("lbox/lbox_open", "statute_classification")
        data_st_plus = load_dataset("lbox/lbox_open", "statute_classification_plus")

        data_st = data_st.shuffle()
        data_st_plus = data_st_plus.shuffle()

        return data_st, data_st_plus


    def preprocess_function(self, examples):
        inputs = examples["facts"]
        targets = examples["statutes"]
        model_inputs = self.tokenizer(inputs, max_length=self.config.max_source_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(targets, max_length=self.config.max_target_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs


    def data_pipeline(self):
        data_st, data_st_plus = self.load_data()
        print("-"*10 + "Data Loaded!" + "-"*10)
        print("-"*10 + "Data Shuffling complete!" + "-"*10)

        data_st = data_st.map(lambda x: {"statutes": [", ".join(label) for label in x["statutes"]]},batched=True,num_proc=1)
        data_st_plus = data_st_plus.map(lambda x: {"statutes": [", ".join(label) for label in x["statutes"]]},batched=True,num_proc=1)
        print("-"*10 + "Finished converting label to text" + "-"*10)

        # concat two datasets
        train_dataset = datasets.concatenate_datasets([data_st["train"], data_st_plus["train"]])
        valid_dataset = datasets.concatenate_datasets([data_st["validation"], data_st_plus["validation"]])
        test_dataset = datasets.concatenate_datasets([data_st["test"], data_st_plus["test"]])

        # intergrate datasets into one DatasetDict
        final_dataset = datasets.DatasetDict({
            "train" : train_dataset,
            "validation" : valid_dataset,
            "test" : test_dataset
        })

        print(final_dataset["train"][0])

        # tokenize and preprocessing for model input
        tokenized_datasets = final_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=final_dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        print(tokenized_datasets)
        print(tokenized_datasets["train"][0])
        print("-"*10 + "Tokenizing Complete!" + "-"*10)

        train_dataloader = DataLoader(
            tokenized_datasets["train"], shuffle=True, collate_fn=default_data_collator, batch_size=self.config.batch_size, pin_memory=True, num_workers=16
        )

        valid_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=default_data_collator, batch_size=self.config.eval_batch_size, pin_memory=True, num_workers=16)
        test_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=default_data_collator, batch_size=self.config.eval_batch_size, pin_memory=True, num_workers=16)
        print("-"*10 + "DataLoader initialized!" + "-"*10)
    

        return train_dataloader, valid_dataloader, test_dataloader
    
