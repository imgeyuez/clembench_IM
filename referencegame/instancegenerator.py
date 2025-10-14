"""
Generate instances for the referencegame

See README in ressources/ on how to generate grids and instances for different languages
and changes across different versions
Currently reads grids_v1.5.json from resources/ (grids didn't change for version 1.6)

Creates instances_{version}_{lang}.json in in/
"""

import logging
import os
import Levenshtein # to calculate distance between grids

from clemcore.clemgame import GameInstanceGenerator

from referencegame.resources.localization_utils import MULTILINGUAL_PATTERNS

logger = logging.getLogger(__name__)

VERSION = "v2.1"
GRIDS = "clembench/referencegame/resources/grids_v2.1.json"
P1_MODE = "strict"  # "liberal"
P2_MODE = "strict"  # "liberal"
# modes relate to regex parsing:
#         Strict parsing mode:
#         - "remainder" should be "" (checked by game master)
#         Liberal parsing mode:
#         - "remainder" doesn't have to be empty (checked by game master)


def generate_samples(grids):
    """
    Generate triplets from grids
    :param grids: list of string grid representations
    :return: list of triplets where the first is the target and the following two are distractors
    """
    samples = []
    # calculate edit distance between grids
    edit_distances = get_distances(grids)
    # select distractors with smallest edit distance
    for target_grid_id in range(len(grids)):
        second_grid_id, third_grid_id = select_distractors(target_grid_id, edit_distances)
        samples.append((grids[target_grid_id], grids[second_grid_id], grids[third_grid_id]))
    return samples


def get_distances(grids):
    """
    Calculate edit distances to select similar distractors
    :param grids: list of string grid representations
    :return: matrix of edit distances with ids corresponding to grids (the full matrix is filled for easier access to distances per grid)
    """
    distances = []
    for i in range(len(grids)):
        i_distances = []
        for j in range(len(grids)):
            if i == j:
                i_distances.append(0)
            else:
                i_distances.append(Levenshtein.distance(grids[i], grids[j]))
        distances.append(i_distances)
    return distances


def select_distractors(target_grid: int, distances: list):
    """
    Select two most similar distractors for the given target
    :param target_grid: id of the target grid in corresponding distance matrix
    :param distances: matrix of distances between grid ids
    :return: ids of two distractor grids
    """
    id1 = None
    id2 = None
    min_distance = 1
    distance_list = distances[target_grid]
    found1 = False
    found2 = False
    while not found2:
        # iterate through list of distances,
        # increasing the distance with every iteration
        # to find the two lowest distances and return their ids
        if min_distance in distance_list:
            if not found1:
                # save id of lowest distance
                id1 = distance_list.index(min_distance)
                found1 = True
            elif id1 != distance_list.index(min_distance):
                # save id of next lowest distance if id1 is already found
                id2 = distance_list.index(min_distance)
                found2 = True
            elif min_distance in distance_list[id1 + 1:]:
                # or save next id of same lowest distance as id1 if it exists
                id2 = distance_list.index(min_distance, id1 + 1)
                found2 = True
            else:
                min_distance += 1
        else:
            min_distance += 1
    return id1, id2
    

def has_adjacent_duplicate_samples(instances):
    ''''
    This function checks if there are adjacent dublicates
    of instances 
    (A target grid which is the same target twice in a row)
    '''
    for i in range(len(instances) - 1):
        if instances[i]["sample_id"] == instances[i + 1]["sample_id"]:
            return True
    return False


class ReferenceGameInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(os.path.dirname(__file__))
        self.lang = None

    def on_generate(self, seed: int, **kwargs):
        """
        Create instances into self.instances
        (Called by super().generate())
        """
        self.lang = kwargs["lang"]

        # load grids
        grids = self.load_json(GRIDS)

        # generate sub experiments
        for grids_group in grids.keys():
            # get triplets
            samples = generate_samples(grids[grids_group])

            player_a_prompt_header = self.load_template(f"resources/initial_prompts/"
                                                        f"{self.lang}/player_a_prompt_header.template")
            player_b_prompt_header = self.load_template(f"resources/initial_prompts/"
                                                        f"{self.lang}/player_b_prompt_header.template")

            all_instances = []
            
            for sample_id, sample in enumerate(samples):
                # create three instances from each triplet, where the target for player 2 is in
                # one of the three possible positions each (selecting one order for the other two)
                for i in [1, 2, 3]:
                    target_grid, second_grid, third_grid = sample

                    instance_data = {}

                    game_instance["player_1_prompt_header"] = player_a_prompt_header.replace('TARGET_GRID', target_grid)\
                                                                                    .replace('SECOND_GRID', second_grid)\
                                                                                    .replace('THIRD_GRID', third_grid)
                    game_instance['player_1_target_grid'] = target_grid
                    game_instance['player_1_second_grid'] = second_grid
                    game_instance['player_1_third_grid'] = third_grid

                    # create order of grids for player 2
                    # extract target grid names from localization_utils
                    targets = MULTILINGUAL_PATTERNS[self.lang]["p2_options"]
                    assert len(targets) == 3

                    if i == 1:
                        first_grid = target_grid
                        target_grid_name = [targets[0][0], targets[0][1], targets[0][2], targets[0][3], "1"] # corresponds to "first"
                    elif i == 2:
                        first_grid = second_grid
                        second_grid = target_grid
                        target_grid_name = [targets[1][0], targets[1][1], targets[1][2], targets[1][3], "2"] # corresponds to "second"
                    elif i == 3:
                        first_grid = third_grid
                        third_grid = target_grid
                        target_grid_name = [targets[2][0], targets[2][1], targets[2][2], targets[2][3], "3"] # corresponds to "third"

                    instance_data["player_2_prompt_header"] = player_b_prompt_header.replace('FIRST_GRID', first_grid)\
                                                                                    .replace('SECOND_GRID', second_grid)\
                                                                                    .replace('THIRD_GRID', third_grid)
                    instance_data['player_2_first_grid'] = first_grid
                    instance_data['player_2_second_grid'] = second_grid
                    instance_data['player_2_third_grid'] = third_grid
                    instance_data['target_grid_name'] = target_grid_name

                    instance_data['player_1_response_pattern'] = self._generate_regex("p1")
                    instance_data['player_2_response_pattern'] = self._generate_regex("p2")

                    instance_data['lang'] = self.lang
                    instance_data['p1_mode'] = P1_MODE
                    instance_data['p2_mode'] = P2_MODE

                    instance_data["sample_id"] = sample_id  # Tag with sample group
                    all_instances.append(instance_data)

            by_target = defaultdict(list)
            for instance in all_instances:
                tgt = instance['player_1_target_grid']
                by_target[tgt].append(instance)

            # Shuffle instances for each target
            for tgt in by_target:
                random.shuffle(by_target[tgt])

            # Now interleave without same-target adjacency
            final_instances = []
            last_target = None

            while any(by_target.values()):
                # Collect available targets that aren't same as last used
                options = [t for t in by_target if by_target[t] and t != last_target]
                
                if not options:
                    # No alternative, force pick any with content to break deadlock
                    options = [t for t in by_target if by_target[t]]

                # Choose a target randomly from allowed options
                target = random.choice(options) # I let it pick them randomly BUT somehow every pattern is asked once in a row with solutions on grid 3, then 2 and then 1 :D so...not so random?
                instance = by_target[target].pop()
                final_instances.append(instance)
                last_target = target

            # add the instances to the experiment
            experiment = self.add_experiment(f"{grids_group}")
            for game_counter, instance in enumerate(final_instances):
                game_instance = self.add_game_instance(experiment, game_counter)
                game_instance.update(instance)

    def _generate_regex(self, player: str):
        """
        Combine language specific content with regex pattern
        The regex uses 3 named groups: tag, response, and remainder
        - model is instructed to start its answer with "tag"
        - the following \s is optional (as asian languages don't use spaces)
        - "content" combines "response" and "remainder"
        - "response" is defined as the first produced paragraph (up to a newline)
        - "remainder" collects all following content

        :param player: string identifier of player ("p1" or "p2")
        :return: regex pattern for given player in the current language
        """
        regex_string = ""
        if player == "p1":
            tag = MULTILINGUAL_PATTERNS[self.lang]["p1_tag"]
            response = ".+"
            regex_string = f'^(?P<tag>{tag}\s*)(?P<content>(?P<response>{response})\n*(?P<remainder>(.|\n)*))'

        elif player == "p2":
            tag = MULTILINGUAL_PATTERNS[self.lang]["p2_tag"]
            # collect answer options for the given language
            # also allow models to output numbers,
            # as we can't do prompt engineering in all languages
            r = ''
            for list in MULTILINGUAL_PATTERNS[self.lang]["p2_options"]:
                for term in list:
                    r = r + term + '|'
            response = f'{r}1|2|3'
            # models are alowed to produce more than the label (but not following a newline)
            # because we can't do prompt-engineering in all languages and models tend to output "grid" or punctuation
            # after the answer from player 2
            regex_string = f'^(?P<tag>{tag}\s*)(?P<content>(?P<response>{response}).*\n*(?P<remainder>(.|\n)*))'

        return regex_string


if __name__ == '__main__':
    # generate language version english:
    for language in MULTILINGUAL_PATTERNS.keys():
        if language == "en":
            ReferenceGameInstanceGenerator().generate(
                filename=f"instances_{VERSION}_{language}.json", seed=42, lang=language)
