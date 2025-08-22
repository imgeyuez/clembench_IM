from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.1-8B-Instruct", # meta-llama/Llama-3.1-8B-Instruct
    max_seq_length = 512,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

data = [
    {
        "prompt": "You are given three grids, where each of them is 5 by 5 in size.\nGrids have empty cells marked with \"▢\" and filled cells marked with \"X\".\nYour task is to generate a referring expression that best describes the target grid while distinguishing it from the two other distractor grids.\nThe first grid is the target grid, and the following two grids are the distractors.\n\nTarget grid:\n\nX X X X X\nX □ □ □ □\nX □ □ □ □\nX □ □ □ □\nX X X X X\n\nDistractor grid 1:\n\nX X X X X\nX X □ □ □\nX X X X X\nX X □ □ □\nX X X X X\n\nDistractor grid 2:\n\nX X X X X\nX □ □ □ □\nX X X □ □\nX □ □ □ □\nX □ □ □ □\n\nInstruction: Describe the target grid.\nGenerate the referring expression starting with the tag \"Expression: \" for the given target grid. Omit any other text.",
        "description": "Expression: The grid that has four X's on the top and bottom and three rows of □'s in the middle, where the first and last rows of □'s are empty.",
        "reward": -1
    },
    
    {
        "prompt": "You are given three grids, where each of them is 5 by 5 in size.\nGrids have empty cells marked with \"▢\" and filled cells marked with \"X\".\nYour task is to generate a referring expression that best describes the target grid while distinguishing it from the two other distractor grids.\nThe first grid is the target grid, and the following two grids are the distractors.\n\nTarget grid:\n\nX X X X X\nX □ □ □ □\nX □ □ □ □\nX □ □ □ □\nX X X X X\n\nDistractor grid 1:\n\nX X X X X\nX X □ □ □\nX X X X X\nX X □ □ □\nX X X X X\n\nDistractor grid 2:\n\nX X X X X\nX □ □ □ □\nX X X □ □\nX □ □ □ □\nX □ □ □ □\n\nInstruction: Describe the target grid.\nGenerate the referring expression starting with the tag \"Expression: \" for the given target grid. Omit any other text.",
        "description": "Expression: The grid that has four X's on the top and bottom and three rows of □'s in the middle, where the first and last rows of □'s are empty.",
        "reward": 1
    },
    
    {
        "prompt": "You are given three grids, where each of them is 5 by 5 in size.\nGrids have empty cells marked with \"▢\" and filled cells marked with \"X\".\nYour task is to generate a referring expression that best describes the target grid while distinguishing it from the two other distractor grids.\nThe first grid is the target grid, and the following two grids are the distractors.\n\nTarget grid:\n\nX X X X X\nX □ □ □ X\nX X X X X\nX X □ □ □\nX X □ □ □\n\nDistractor grid 1:\n\nX X X X X\nX X □ □ □\nX X X X X\nX X □ □ □\nX X X X X\n\nDistractor grid 2:\n\nX X X X X\nX □ □ □ □\nX X X □ □\nX □ □ □ □\nX □ □ □ □\n\nInstruction: Describe the target grid.\nGenerate the referring expression starting with the tag \"Expression: \" for the given target grid. Omit any other text.",
     
        "description": "Expression: The grid that has a row with three X's in the top row, a row with three X's in the third row, and two rows with two □'s in the fourth and fifth rows.",
     
        "reward": -1
    },
    
    {
        "prompt": "You are given three grids, where each of them is 5 by 5 in size.\nGrids have empty cells marked with \"▢\" and filled cells marked with \"X\".\nYour task is to generate a referring expression that best describes the target grid while distinguishing it from the two other distractor grids.\nThe first grid is the target grid, and the following two grids are the distractors.\n\nTarget grid:\n\nX X X X X\nX X □ □ □\nX X X X X\nX X □ □ □\nX X X X X\n\nDistractor grid 1:\n\nX X X X X\nX X □ X X\nX X X X X\nX X □ X X\nX X X X X\n\nDistractor grid 2:\n\nX X X X X\nX □ □ □ X\nX X X X X\nX X □ □ □\nX X □ □ □\n\nInstruction: Describe the target grid.\nGenerate the referring expression starting with the tag \"Expression: \" for the given target grid. Omit any other text.",
     
        "description": "Expression: The grid that has two rows of □ in the middle, with the □ in the second row being surrounded by X on both sides.",
        "reward": -1
    }
]

from datasets import Dataset

dataset = Dataset.from_list(data)

print(dataset)

from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    prompt = example["prompt"]
    response = example["description"]
    reward = example["reward"]

    # Tokenize separately
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    response_tokens = tokenizer(response, add_special_tokens=False)

    input_ids = prompt_tokens["input_ids"] + response_tokens["input_ids"]
    attention_mask = [1] * len(input_ids)

    # Mask the prompt tokens — only apply loss to response
    labels = [-100] * len(prompt_tokens["input_ids"]) + response_tokens["input_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "reward": reward,  # single reward per example
    }

tokenized_dataset = dataset.map(tokenize)

print(tokenized_dataset)

from unsloth import FastLanguageModel
from trl import SFTTrainer
from trl.trainer import SFTConfig

import wandb

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = tokenized_dataset,
    max_seq_length = 2048,
    dataset_text_field = None,  # We manually prepared input_ids/labels
    packing = False,
    loss_type = "reward",             # Enable REINFORCE loss
    reward_field = "reward",          # Specify where reward comes from
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        max_steps = 20,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb",
        run_name = 'whyareurunnin'
    ),
)

trainer.train()
