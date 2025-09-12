import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class GeneratorDataset(Dataset):
    def __init__(self, generator_function, target_generator, num_samples=500, deterministic=True):
        super().__init__()

        self.num_samples = num_samples
        self.generator_function = generator_function
        self.target_generator = target_generator

        if deterministic:
            self.samples = self.generator_function(num_samples)
            self.targets = self.target_generator(self.samples)
        else:
            self.samples = None
            self.targets = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.samples is not None:
            return {"y": self.samples[idx], "x": self.targets[idx]}
        else:
            y = self.generator_function(num_samples=1)[0]
            x = self.target_generator(y)
            return {"y": y, "x": x}

    def get_dataloader_input(self, batch_size, num_workers=1):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=False)

    def get_dataloader_target(self, batch_size,num_workers):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
