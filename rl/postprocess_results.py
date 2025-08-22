"""
Since I wasn't able to correctly incorporate the acceptance of the 
answers "1st", "2nd" and "3rd" within the master.py file, 
this script is a post-procession step, where I iterate through the
results and if a game outcome is "parsed_wrong", it checks 
if the content is either "1s", "2n", or "3r", and if so, 
if the next letter would have been "t", "d", "d", respectively.
If so, the game instance will be turned into a success.
"""

import os
import json

# Path to the results directory
RESULTS_PATH = r".\results"
GAME = "referencegame"
OUTPUT_PATH = r"C:\Users\imgey\Desktop\MASTERS\MASTER_POTSDAM\SoSe25\IM\codespace\data"

# data for comprehension, generation, comprehension datasharing, generation datasharing, human data
dl, ds, dl_DS, ds_DS = list(), list(), list(), list()
d_human = {
    "generation": list(),
    "comprehension": list()
}

print("Start turning instances into tuple datapoints.")

# Iterate through all experiments
for experiment in os.scandir(RESULTS_PATH):
    if not experiment.is_dir():
        continue  # Skip files

    current_path = os.path.join(experiment.path, GAME)

    if not os.path.isdir(current_path):
        continue  # Skip if GAME folder doesn't exist

    # Iterate through all game modes
    for game_mode in os.scandir(current_path):
        if not game_mode.is_dir():
            continue  # Skip files like experiment.json

        # Iterate through all episodes
        for episode in os.scandir(game_mode.path):
            if not episode.is_dir():
                continue  # Skip files

            print(f"Episode path: {episode.path}")

            for file in os.scandir(episode.path):
                if file.name.endswith("experiment.json"):
                    continue
                
                # go into the iteractions.json files 
                if file.name == "interactions.json":
                    with open(file.path, "r") as f:
                        interactions = json.load(f)
                    
                    instance_file = os.path.join(episode.path, "instance.json")
                    
                    with open(instance_file, "r") as f:
                        instance = json.load(f)
                    
                    target_grid_names = instance["target_grid_name"]
                    turns = interactions["turns"][0]

                    if len(turns) == 6 and turns[-1]["type"] == "parsed_wrong":

                        corrections = {
                            "1s": "1st",
                            "2n": "2nd",
                            "3r": "3rd",
                        }

                        for key, value in corrections.items():
                            if turns[-1]["content"] == key and value in target_grid_names:
                                
                                turns[-1]["content"] = value
                                turns[-1]["type"] = "parse_correct"

                                with open(file.path, "w") as f:
                                    json.dump(interactions, f, indent=2)