# Installation
See the original [Jaxrl repo](https://github.com/ikostrikov/jaxrl) for instructions.

# Usage

For continuous control from pixels using DrQ-v2:

```bash
MUJOCO_GL=egl python train_pixels.py --env_name=quadruped-run --save_dir=./tmp/
```

To track experiments with Weights and Biases, append `--track` to the above command.

Tune the hyperparameters in examples/configs/drq_v2.py



# Results
Verified performance in quadruped-run, quadruped-walk, acrobot-swingup, cheetah-run, and reacher-hard.

Wandb project opening soon....
