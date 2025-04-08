# CDS521-project
```markdown
# Car Racing Proximal Policy Optimization (PPO) Implementation

**Goal**: Implement Proximal Policy Optimization (PPO) for the CarRacing-v0 environment.

---

## Table of Contents
- [Environment Setup](#environment-setup)
- [Training the Agent](#training-the-agent)
- [Evaluating the Model](#evaluating-the-model)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Environment Setup

### 1. Prerequisites
- **Python 3.7** (required for TensorFlow 1.x compatibility)  
  Download: [Python 3.7.10](https://www.python.org/downloads/release/python-3710/)
- **pip** (latest version recommended)

### 2. Create a Virtual Environment
```bash
# For Windows
python -m venv car_env
car_env\Scripts\activate

# For Linux/macOS
python3.7 -m venv car_env
source car_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install tensorflow==1.15.5 gym==0.17.3 numpy==1.19.5 opencv-python pyglet Box2D scipy

# For Chinese users (use Tsinghua mirror)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.15.5 gym==0.17.3 numpy==1.19.5 opencv-python pyglet Box2D scipy
```

### 4. Verify Installation
```bash
python -c "import tensorflow as tf; print(tf.__version__)"  # Should output 1.15.5
python -c "import gym; print(gym.__version__)"             # Should output 0.17.3
```

---

## Training the Agent

### Start Training
```bash
python train.py \
  --num_envs 16 \
  --horizon 128 \
  --batch_size 128 \
  --num_epochs 10 \
  --initial_lr 3e-4 \
  --discount_factor 0.99 \
  --gae_lambda 0.95 \
  --ppo_epsilon 0.2
```

**Key Arguments**:
- `--num_envs`: Number of parallel environments (default: 16).
- `--horizon`: Steps per environment per rollout (default: 128).
- `--save_interval`: Save model every N steps (default: 1000).

---

## Evaluating the Model

### 1. Monitor Training Metrics with TensorBoard
```bash
tensorboard --logdir=./logs/CarRacing-v0
```
Access metrics at `http://localhost:6006`:
- `eval_avg_reward`: Average reward per evaluation.
- `loss_policy`: Policy network loss.

### 2. View Evaluation Videos
- Videos are saved in `./videos/CarRacing-v0/`.
- Use a video player (e.g., VLC) to open files like `step200.avi`.

---

## Project Structure
```
.
├── train.py             # Main training script
├── ppo.py               # PPO algorithm implementation
├── utils.py             # Utilities (frame stacking, GAE calculation)
├── CarRacing-run_random_agent.py  # Random agent demo
├── car_racing.py        # Custom CarRacing-v0 environment
├── models/              # Saved model checkpoints
├── logs/                # TensorBoard logs
└── videos/              # Recorded evaluation episodes
```

---

## Key Features
- **Frame Preprocessing**:  
  - Crop to 84x84, grayscale, and normalize input frames.
  - Frame stacking (4-frame history) for temporal information.
- **PPO Implementation**:
  - Clipped surrogate objective for stable policy updates.
  - Generalized Advantage Estimation (GAE) for advantage calculation.
  - Entropy regularization to encourage exploration.

---

## Troubleshooting

### 1. Installation Issues
- **TensorFlow Compatibility**:  
  Ensure Python 3.7 is used. For GPU support, install CUDA 10.0 and cuDNN 7.4.
- **Box2D Dependency**:  
  Install SWIG first: `pip install swig`, then retry `pip install gym[box2d]`.

### 2. Training Issues
- **Low Reward**:  
  Adjust hyperparameters (`--ppo_epsilon`, `--gae_lambda`).
- **Memory Errors**:  
  Reduce `--num_envs` or `--batch_size`.

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Note**: For detailed algorithm explanations, refer to the [docs](docs/) directory.  
**Acknowledgments**: Based on OpenAI Gym and TensorFlow 1.x.
``` 

### Key Improvements Over Original README:
1. **Clearer Environment Setup**: Explicit Python 3.7 requirement and installation commands.
2. **Step-by-Step Instructions**: Added verification steps and troubleshooting.
3. **Evaluation Guidance**: Explained TensorBoard usage and video recording.
4. **Project Structure**: Mapped file/directory purposes for clarity.
5. **Hyperparameter Context**: Clarified key training arguments.
