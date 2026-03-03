# QSIM: Mitigating Overestimation in Multi-Agent Reinforcement Learning via Action Similarity Weighted Q-Learning


## 📂 Code Structure

The codebase is organized into two separate directories based on the benchmark environments:

*   `pymarl/`: Based on the original [PyMARL](https://github.com/oxwhirl/pymarl), used for **SMAC** experiments.
*   `epymarl/`: Based on [EPyMARL](https://github.com/uoe-agents/epymarl), used for **SMACv2**, **MPE**, and **Matrix Games**.

## ⚙️ Installation & Requirements

Since the two sub-directories rely on slightly different dependencies, please install the requirements for the specific environment you wish to run.

### 1. Prerequisites
*   Python 3.8
*   PyTorch 2.1.0
*   StarCraft II (4.10) for SMAC/SMACv2

### 2. StarCraft II Setup
Run the included installation script to download SC2 and the required maps:
```bash
bash install_sc2.sh
```
Or set the `SC2PATH` environment variable manually.

### 3. Python Dependencies
Install the required packages for each codebase:

**For SMAC (PyMARL):**
```bash
cd pymarl
pip install -r requirements.txt
```

**For SMACv2 / MPE / Matrix Games (EPyMARL):**
```bash
cd epymarl
pip install -r requirements.txt
pip install -r env_requirements.txt
```

---

## 🚀 Run Experiments

The core algorithm config is `qsim_qmix`. Below are the commands to reproduce the experiments in the paper.

### 1. SMAC (StarCraft Multi-Agent Challenge)

The SMAC experiments are conducted within the `pymarl` directory.

**Command Template:**
```bash
cd pymarl
python src/main.py --config=qsim_qmix --env-config=sc2 with env_args.map_name=<MAP_NAME>
```

**Examples:**
```bash
# Run on MMM2
cd pymarl
python src/main.py --config=qsim_qmix --env-config=sc2 with env_args.map_name=MMM2
```

---

### 2. SMACv2, MPE, and Matrix Games

These experiments are conducted within the `epymarl` directory. **Ensure you switch directories before running these commands.**

#### A. SMACv2
SMACv2 uses procedural generation. The map name typically defines the unit distribution (e.g., `terran_5_vs_5`).

**Command Template:**
```bash
cd epymarl
python src/main.py --config=qsim_qmix --env-config=sc2v2 with env_args.map_name="<MAP_NAME>"
```

**Example:**
```bash
# Run on Terran 5 vs 5
cd epymarl
python src/main.py --config=qsim_qmix --env-config=sc2v2 with env_args.map_name="terran_5_vs_5"
```

#### B. MPE (Multi-Agent Particle Environments)
We use the PettingZoo implementation wrapped via `gymma`.

**Simple Tag:**
```bash
cd epymarl
python src/main.py --config=qsim_qmix --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag"
```

**Simple Adversary:**
```bash
cd epymarl
python src/main.py --config=qsim_qmix --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-adversary-v3" env_args.pretrained_wrapper="PretrainedAdversary"
```

#### C. Matrix Games
We use the non-state version of the Climbing Game to test convergence.

**Climbing Game:**
```bash
cd epymarl
python src/main.py --config=qsim_qmix --env-config=gymma with env_args.time_limit=25 env_args.key="matrixgames:climbing-nostate-v0"
```

---

## 📝 Hyperparameters

The specific hyperparameters for QSIM are defined in `src/config/algs/qsim_qmix.yaml` in both directories.

For a full list of hyperparameters and experimental settings, please refer to the Appendix of our paper.

## Cite QSIM

If you use QSIM in your research, please cite the following paper:

Yuanjun Li, Bin Zhang, Hao Chen, Zhouyang Jiang, Dapeng Li, and Zhiwei Xu. QSIM: Mitigating Overestimation in Multi-Agent Reinforcement Learning via Action Similarity Weighted Q-Learning, arXiv preprint arXiv:2602.22786, 2026.

In BibTeX format:

```tex
@article{li2026qsim,
  title={QSIM: Mitigating Overestimation in Multi-Agent Reinforcement Learning via Action Similarity Weighted Q-Learning},
  author={Li, Yuanjun and Zhang, Bin and Chen, Hao and Jiang, Zhouyang and Li, Dapeng and Xu, Zhiwei},
  journal={arXiv preprint arXiv:2602.22786},
  year={2026}
}
