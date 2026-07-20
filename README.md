# ADAPT
Code for paper ADAPT

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/xuyangthu88/ADAPT.git
cd ADAPT
```

### 2. Create conda environment

```bash
conda create -n adapt python=3.10
conda activate adapt
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
### 4. Editable install packages
```bash
pip install -e .  # install cleanrl
cd diffuser
pip install -e .  # install diffuser
```

### 5. (Optional) Install EnergyPlus

For Sinergym experiments, install EnergyPlus 23.1+ and set:

```bash
export ENERGYPLUS_HOME={your energyplus path}
export PATH=$ENERGYPLUS_HOME:$PATH
export PYTHONPATH=$ENERGYPLUS_HOME:$PYTHONPATH
export EPLUS_PATH={your energyplus path}
```

## Training

### Train the diffusion IEWM

```bash
python scripts/train_iewm.py \
  --config configs/iewm_diffusion.yaml
```

### Train RL with the world model

```bash
python scripts/train_adapt.py \
  --config configs/adapt_bdq.yaml
```

### Evaluate a trained policy

```bash
python scripts/evaluate.py \
  --checkpoint results/adapt/model.pt
```
