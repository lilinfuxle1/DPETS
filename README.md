# Dropout-based Probabilistic Ensembles with Trajectory Sampling

Code to reproduce the experiments in [Practical Probabilistic Model-based Reinforcement Learning by Integrating Dropout Uncertainty and Trajectory Sampling](https://ieeexplore.ieee.org/document/10721239). This article has been accepted for publication in IEEE Transactions on Neural Networks and Learning Systems.

![method.png](https://raw.githubusercontent.com/mrjun123/DPETS/main/images/method.png)

Please feel free to contact us regarding to the details of implementing DPETS. (Wenjun Huang: wj.huang1@siat.ac.cn Yunduan Cui: cuiyunduan@gmail.com)
## Requirements

1. Prepare the Project Directory
Clone or download the project code to your local machine and navigate to the project root directory:

```bash
git clone https://github.com/your-repo/dpets.git
cd dpets
```
2. Create and Activate a Virtual Environment (Recommended)
To avoid dependency conflicts, it's recommended to use a virtual environment:

```bash
conda create -n dpets python=3.10
conda activate dpets
```
 3. PyTorch Installation (with CUDA Compatibility)
PyTorch automatically includes compatible versions of CUDA runtime, cuDNN, cuBLAS, and other dependencies. There is no need to manually install nvidia-* packages:
In this experiment, we use PyTorch torch==2.9.0 with CUDA 12.x.
```bash
pip install torch torchvision torchaudio
```
4. Install Gymnasium and mujoco
In this experiment, we use mujoco==3.3.7 and gymnasium==1.2.1
```bash
pip install gymnasium[all]
pip install mujoco
```
5. Install Other Dependencies
```bash
pip install -r requirements.txt
```

## Running Experiments

Experiment for a specific configuration can be run using:

```python
python main.py --config cartpole
```

The specific configuration file is located in the `configs` directory and the default configuration file can be located in the root directory `default_config.json` was found, which allows you to modify the experimental parameters.

## Logging

We use Tensorboard to record experimental data, you can view runs with:

```python
tensorboard --logdir ./runs/ --port=6006 --host=0.0.0.0
```

## Reference
```
@ARTICLE{10721239,
  author={Huang, Wenjun and Cui, Yunduan and Li, Huiyun and Wu, Xinyu},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Practical Probabilistic Model-Based Reinforcement Learning by Integrating Dropout Uncertainty and Trajectory Sampling}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  keywords={Uncertainty;Probabilistic logic;Predictive models;Trajectory;System dynamics;Fitting;Stability analysis;Computational modeling;Robustness;Learning systems;Predictive models;reinforcement learning (RL);robot learning},
  doi={10.1109/TNNLS.2024.3474169}}
```
