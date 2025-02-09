import os
import torch
import datasets
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    GenerationConfig
)
from loss import CustomTrainer
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_id = "meta-llama/Llama-2-7b-chat-hf"
max_length = 512
device_map = "auto"
batch_size = 128
micro_batch_size = 32
gradient_accumulation_steps = batch_size // micro_batch_size #4

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,   # load the model into memory using 4-bit precision
    bnb_4bit_use_double_quant=True, # use double quantition
    bnb_4bit_quant_type="nf4", # use NormalFloat quantition
    bnb_4bit_compute_dtype=torch.bfloat16 # use hf for computing when we need
)

# load model from huggingface
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    device_map=device_map
)

# load tokenizer from huggingface
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

model.resize_token_embeddings(len(tokenizer))

tokenizer.padding_side = "right"
      
def print_number_of_trainable_model_parameters(model):
  trainable_model_params = 0
  all_model_params = 0
  for _, param in model.named_parameters():
    all_model_params += param.numel()
    if param.requires_grad:
      trainable_model_params += param.numel()
  print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
  return trainable_model_params

ori_p = print_number_of_trainable_model_parameters(model)

model = prepare_model_for_kbit_training(model)
'''
- r, the dim of the low_rank matrices
- lora_alpha, scaling factor, the weight is scaled by lora_alpha/r, 
  the higher value assigns more weight to the LoRA activations
- target_modules: default is "q_proj", "v_proj"
- bias, the recommend setting bias to None first, and then lora_only, before trying all.
'''
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

peft_p = print_number_of_trainable_model_parameters(model)

def generate_prompt(instruction=None, question=None, choices=None, answer=None, explanation=None):
    if question is None or choices is None:
        return f"Answer: {answer} \n\n[Explanation] {explanation}"
    else:
        return f"{instruction}\n\n[Question]\n{question}\n\n[Choices]\n{choices}\n\n[Response]\nAnswer: {answer} \n\n[Explanation] {explanation}"

def generate_and_tokenize_prompt(batch, tokenizer):
    instruction = "Your task is to select the correct answer and provide a detailed explanation for why it is the correct choice."
    max_length = 512
    
    full_prompts = [
        generate_prompt(instruction, question, choices, answer, explanation)
        for question, choices, answer, explanation in zip(batch["question"], batch["choices"], batch["answer"], batch["ground_truth"])
    ]
    labels_only = [
        generate_prompt(None, None, None, answer, explanation)
        for answer, explanation in zip(batch["answer"], batch["ground_truth"])
    ]
    
    tokenized_full_prompts = tokenizer(full_prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    tokenized_labels_only = tokenizer(labels_only, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    
    input_ids = tokenized_full_prompts['input_ids']
    attention_mask = tokenized_full_prompts['attention_mask']
    actual_sequence_lengths = attention_mask.sum(dim=1)
    labels = torch.full_like(input_ids, -100)
    for i in range(labels.size(0)):
        seq_len = actual_sequence_lengths[i].item()
        ans_len = tokenized_labels_only['attention_mask'][i].sum().item()
        ans_start = seq_len - ans_len
        labels[i, ans_start:seq_len] = tokenized_labels_only['input_ids'][i, :ans_len]
    #s_scores = torch.tensor(batch['faithfulness'], dtype=torch.float32).to(input_ids.device)
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        #"s_score": s_scores,
        "s_score": batch["faithfulness"], 
    }
    return result
    
# def custom_collate_fn(batch):
#     print("Custom collate fn called")  
#     input_ids = torch.stack([torch.tensor(item['input_ids'], dtype=torch.long) for item in batch])
#     attention_mask = torch.stack([torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch])
#     labels = torch.stack([torch.tensor(item['labels'], dtype=torch.long) for item in batch])
#     s_score = torch.tensor([item['s_score'] for item in batch], dtype=torch.long)

#     return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 's_score': s_score}
def custom_collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids'], dtype=torch.long) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch])
    labels = torch.stack([torch.tensor(item['labels'], dtype=torch.long) for item in batch])
    s_scores = [item['s_score'] for item in batch]

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 's_scores': s_scores}

dataset = datasets.load_dataset("csv", data_files="merged_output.csv", split="train[:6000]")
split_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
train_data = split_dataset['train']
val_data = split_dataset['test']

train_data = train_data.map(lambda batch: generate_and_tokenize_prompt(batch, tokenizer), batched=True)
val_data = val_data.map(lambda batch: generate_and_tokenize_prompt(batch, tokenizer), batched=True)

print(train_data[0])
train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_data, batch_size=batch_size, collate_fn=custom_collate_fn)

args = Seq2SeqTrainingArguments(
    output_dir='llama-7b-ecqa-IR',
    num_train_epochs=10,
    fp16=True,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    per_device_train_batch_size=micro_batch_size,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    group_by_length=False,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    disable_tqdm=False,
    evaluation_strategy="steps",
    eval_steps=0.1,
)

model.to(device)
trainer = CustomTrainer(
    model=model,
    # train_dataset=train_data,
    # eval_dataset=val_data,
    args=args,
    data_collator=DataCollatorForSeq2Seq(
      tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)
trainer.train_dataloader = train_dataloader
trainer.eval_dataloader = val_dataloader


print("CustomTrainer created.")

model_save_path = "llama-7b-ecqa-IR"

model.config.use_cache = False
IS_RESUME = False

if IS_RESUME:
  trainer.train(f'{model_save_path}/checkpoint-last')
else:
  trainer.train()
model.save_pretrained(model_save_path)
print('model train is finished')