import os
import torch

class ModelLoader:
    def __init__(self, models_name, device, part_list) -> None:
        self.model_name = models_name
        self.DEVICE = device
        self.part_list = part_list

    def load_model(self):
        print(f"loading model... => {self.model_name}")
        model_part = {}
        for part in self.part_list:
            model_part[part] = self._load_model(self.model_name, part)
        print("<---finished loading--->")

        return model_part

    def _load_model(self, model_name, part):
        path = os.path.join("model_files", model_name, part)
        model = torch.load(path + "/best.pth",map_location = self.DEVICE).to(self.DEVICE)
        model.eval()

        return model