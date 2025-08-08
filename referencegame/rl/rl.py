import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

"""Example
"{"meta": {"game_name": "referencegame", "experiment_name": "letter_grids", "game_id": 0, "dialogue_pair": "human-t0.0--human-t0.0", "clem_version": "2.4.3"}, "players": {"GM": {"game_role": "Game Master", "model_name": "programmatic"}, "Player 1": {"game_role": "Instruction Giver", "model_name": "human"}, "Player 2": {"game_role": "Instruction Follower", "model_name": "human"}}, "turns": [[{"from": "GM", "to": "Player 1", "timestamp": "2025-06-29T12:47:02.268441", "action": {"type": "send message", "content": "You are given three grids, where each of them is 5 by 5 in size.\nGrids have empty cells marked with \"▢\" and filled cells marked with \"X\".\nYour task is to generate a referring expression that best describes the target grid while distinguishing it from the two other distractor grids.\nThe first grid is the target grid, and the following two grids are the distractors.\n\nTarget grid:\n\nX X X X X\nX □ □ □ □\nX X X □ □\nX □ □ □ □\nX □ □ □ □\n\nDistractor grid 1:\n\nX X X X X\nX □ □ □ X\nX X X X X\nX X □ □ □\nX X □ □ □\n\nDistractor grid 2:\n\nX X X X X\nX □ □ □ □\nX □ □ □ □\nX □ □ □ □\nX X X X X\n\nInstruction: Describe the target grid.\nGenerate the referring expression starting with the tag \"Expression: \" for the given target grid. Omit any other text.", "label": "context"}}, {"from": "Player 1", "to": "GM", "timestamp": "2025-06-29T12:47:07.339719", "action": {"type": "get message", "content": "Expression: F", "label": "response"}}, {"from": "GM", "to": "GM", "timestamp": "2025-06-29T12:47:07.340538", "action": {"type": "parse", "content": "Expression: F"}}, {"from": "GM", "to": "Player 2", "timestamp": "2025-06-29T12:47:07.340538", "action": {"type": "send message", "content": "You are given three grids, where each of them is 5 by 5 in size.\nGrids have empty cells marked with \"▢\" and filled cells marked with \"X\".\nYou are also given a referring expression that describes one of the given grids.\nYour task is to select the grid that matches the given referring expression.\nGenerate only the number (in text) of the grid that matches the given expression by selecting first, second, or third.\n\nFirst:\n\nX X X X X\nX □ □ □ □\nX X X □ □\nX □ □ □ □\nX □ □ □ □\n\nSecond:\n\nX X X X X\nX □ □ □ X\nX X X X X\nX X □ □ □\nX X □ □ □\n\nThird:\n\nX X X X X\nX □ □ □ □\nX □ □ □ □\nX □ □ □ □\nX X X X X\n\nExpression: F\nQuestion: Which grid does the expression refer to?\nStart with the tag \"Answer: \", followed by your selection. Omit any other text.\n", "label": "context"}}, {"from": "Player 2", "to": "GM", "timestamp": "2025-06-29T12:49:09.276796", "action": {"type": "get message", "content": "Answer: First", "label": "response"}}, {"from": "GM", "to": "GM", "timestamp": "2025-06-29T12:49:09.277240", "action": {"type": "parse_wrong", "content": "first"}}]]}"
"""

LEARN_TYPE = "rl" # alternative: sft
MODELNAME = "meta-llama/Llama-3.1-8B"

# 1. Preprocess the data 

# data_path = ""

# with open(data_path, "r", encoding="UTF-8") as f:
#     data = json.load(f)
data = [
    {"prompt": "bla bla bla",
     "description": "ha ha ha ah",
     "reward": -1},
    {"prompt": "shubidu",
     "description": "dadadu",
     "reward": -1},
    {"prompt": "bla bla bla",
     "description": "ha ha ha",
     "reward": 1},
    {"prompt": "eins zwei drei",
     "description": "vier fünf sechs",
     "reward": 1},
    {"prompt": "du bist ein Schuh",
     "description": "Ich weiß",
     "reward": -1},
]

train, val, test = list(), list(), list()

data_lengt = len(data)
train_length = data_lengt/100*75
val_length = data_lengt/100*15
test_length = data_lengt/100*10

for i, datapoint in enumerate(data):
    if i > train_length:
        train.append(
            {
                "prompt": datapoint["prompt"],
                "response": data["description"],
                "reward": datapoint["reward"]})
    elif train_length <= i >= test_length:
        val.append(
            {
                "prompt": datapoint["prompt"],
                "response": data["description"],
                "reward": datapoint["reward"]})
    elif i > val_length:
        test.append(
            {
                "prompt": datapoint["prompt"],
                "response": data["description"],
                "reward": datapoint["reward"]})


# 2. prepare for training model

# Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained(MODELNAME) # replace with llama model
tokenizer = AutoTokenizer.from_pretrained(MODELNAME) # replace with llama model

# Convert data into huggingface data format
hf_dataset = Dataset.from_list(data)

# Define Data Collator (important to only compute loss on responses)
data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template="Expression:") # might need adaptation

# Initialize Trainer with loss_type="reward"
trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    max_seq_length=512,
    args={
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "logging_steps": 10,
        "save_steps": 500,
        "learning_rate": 5e-6,
    },
    loss_type="reward",  # activates REINFORCE-style loss
)

# Run training
trainer.train()
