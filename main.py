import hydra
from omegaconf import DictConfig
from tools.solver import Solver


@hydra.main(config_path="../config", config_name="main", version_base="1.3")
def main(
      cfg: DictConfig
):
        solver = Solver(
            cfg = cfg
    )
        solver.fit(cfg)


if __name__ == "__main__":
    main()