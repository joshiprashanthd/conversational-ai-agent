import os
import json

source_path = "./intervention.json"
target_path = "./intervention-migrate.json"


def migrate():
    with open(source_path, "r") as source_file, open(target_path, "w") as target_file:
        result = []
        json_data = json.load(source_file)

        interventions = json_data[0]["interventions"]

        for obj in interventions:
            new_obj = {
                "activity": obj["description"],
                "type": obj["type"],
                "reasoning": obj["reasoning"],
                "howto": obj["howto"],
                "affecting_params": [
                    {
                        "name": "ease of falling asleep",
                        "effectiveness": obj["effectiveness"],
                    }
                ],
            }

            result.append(new_obj)

        json.dump(result, target_file, indent=4, ensure_ascii=False)


migrate()
