from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    seed: int = 42
    # DO NOT touch theese
    data_path: Path = Path("data/allSyntheticData.RData")
    output_path: Path = Path("data") # Defaulted to HIV dataset
    test_size: float = 0.2

#______________________________________________________
    # CVAE and GANs hyperparameters
    z_dim: int = 16
    hidden: int = 128
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    x_transform: str = "log1p" # none | log1p (distribution of X) 

    # CVAE specific
    beta: float = 0.5
    decoder_noise: float = 0
#____________________   __________________________________
    # Keys inside .RData
    x_key: str = "x"
    y_key: str = "y"



    def transform_name(self) -> str:
        return (self.x_transform or "none").strip().lower()

    @property
    def out_dir(self) -> Path:
        return self.output_path / self.transform_name()

    def ensure_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)