"""
LLaVA-MLA Finetuning Script
Full finetuning for MLA-converted LLaVA models
"""
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from datasets import load_dataset
from PIL import Image
import json

# Import customize model loader
from .load import load_llava_with_optional_mla

@dataclass
class ModelArguments:
    model_path: str = field(metadata={"help": "Path to MLA-converted model"})
    vision_tower: Optional[str] = field(default=None)
    
    
@dataclass
class DataArguments:
    data_path: str = field(default="liuhaotian/LLaVA-Instruct-150K")
    image_folder: str = field(default="data/coco2017/train2017")
    image_aspect_ratio: str = field(default="square")
    sample_ratio: float = field(default=0.01, metadata={"help": "Ratio of dataset to use (0.0-1.0)"})


@dataclass
class TrainingArgs(TrainingArguments):
    output_dir: str = field(default="./checkpoints/llava-mla-ft")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    gradient_checkpointing: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    report_to: str = field(default="none")


class LLaVADataset(Dataset):
    """LLaVA Instruction Dataset"""
    
    def __init__(self, data_path, image_folder, processor, sample_ratio=1.0):
        super().__init__()
        self.processor = processor
        self.image_folder = image_folder
        
        # Load dataset
        print(f"Loading dataset from {data_path}...")
        if os.path.isfile(data_path):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            # Load from HF with proper configuration
            try:
                # Try loading as JSON dataset
                dataset = load_dataset(
                    "json",
                    data_files={"train": "hf://datasets/liuhaotian/LLaVA-Instruct-150K/llava_instruct_150k.json"},
                    split='train'
                )
                self.data = list(dataset)
            except Exception as e:
                print(f"Failed to load from HF: {e}")
                print("Attempting direct download...")
                # Fallback: download the file directly
                import requests
                url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"
                print(f"Downloading from {url}...")
                response = requests.get(url)
                self.data = response.json()
        
        print(f"Loaded {len(self.data)} examples")
        
        if sample_ratio < 1.0 and sample_ratio > 0.0:
            import random
            random.seed(42)
            random.shuffle(self.data)
            limit = int(len(self.data) * sample_ratio)
            self.data = self.data[:limit]
            print(f"Sampling {sample_ratio*100:.1f}% of data: Kept {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load image
        image_file = sample.get('image', '')
        if image_file:
            image_path = os.path.join(self.image_folder, image_file)
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Return blank image on error
                image = Image.new('RGB', (224, 224), color='white')
        else:
            image = Image.new('RGB', (224, 224), color='white')
        
        # Build conversation
        conversations = sample.get('conversations', [])
        
        # Format: alternating human/gpt turns
        text = ""
        for conv in conversations:
            role = conv.get('from', '')
            value = conv.get('value', '')
            
            if role == 'human':
                text += f"USER: {value}\n"
            elif role == 'gpt':
                text += f"ASSISTANT: {value}\n"
        
        return {
            'image': image,
            'text': text,
            'conversations': conversations
        }


def collate_fn(batch, processor):
    """Custom collate function for batching"""
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    conversations = [item['conversations'] for item in batch]
    
    # Process images and texts
    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Create labels (same as input_ids, but -100 for padding)
    labels = inputs['input_ids'].clone()
    
    # Mask out prompt tokens (only compute loss on assistant responses)
    for i, convs in enumerate(conversations):
        target_ids = labels[i]
        
        # Find ASSISTANT tokens and only keep loss on those
        text = texts[i]
        
        # Simple heuristic: mask everything before "ASSISTANT:"
        assistant_start = text.find("ASSISTANT:")
        if assistant_start != -1:
            prompt_text = text[:assistant_start]
            prompt_tokens = processor.tokenizer(
                prompt_text, 
                add_special_tokens=False
            )['input_ids']
            prompt_len = len(prompt_tokens)
            target_ids[:prompt_len] = -100
        
        labels[i] = target_ids
    
    inputs['labels'] = labels
    
    return inputs


class LLaVATrainer(Trainer):
    """Custom trainer for LLaVA"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for vision-language model"""
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgs))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load processor and model using your MLA loader
    print(f"Loading MLA-converted model from {model_args.model_path}")
    
    model, processor = load_llava_with_optional_mla(
        model_name_or_path=model_args.model_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        device="cuda", 
        trust_remote_code=True,
        local_files_only=False,
    )
    
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")
    if hasattr(model, 'language_model'):
        print(f"Language model type: {type(model.language_model)}")
    
    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, 'language_model'):
            model.language_model.gradient_checkpointing_enable()
    
    # Prepare dataset
    train_dataset = LLaVADataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        processor=processor,
        sample_ratio=data_args.sample_ratio
    )
    
    # Create data collator
    def data_collator(batch):
        return collate_fn(batch, processor)
    
    # Initialize trainer
    trainer = LLaVATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving final model to {training_args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    main()