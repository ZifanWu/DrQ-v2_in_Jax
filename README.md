This repository implements [DrQ-v2](https://arxiv.org/abs/2107.09645) for pixel-based observation tasks on DMControl.

# Installation
See the original [Jaxrl repo](https://github.com/ikostrikov/jaxrl) for instructions.

# Usage

```bash
cd examples
MUJOCO_GL=egl python train_pixels.py --env_name=quadruped-run --save_dir=./tmp/
```

To track experiments with Weights and Biases, append `--track` to the above command.

Tune the hyperparameters in examples/configs/drq_v2.py



# Results
Verified performance in quadruped-run, quadruped-walk, acrobot-swingup, cheetah-run, and reacher-hard.

Wandb project opening soon....
