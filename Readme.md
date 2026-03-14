
# Double Inverted Pendulum - Reinforcement Learning

This repository implements a custom reinforcement learning (RL) environment for a **Double Inverted Pendulum** using `pymunk` (for 2D physics simulation) and `pygame` (for rendering). It trains a Proximal Policy Optimization (PPO) agent using `stable-baselines3` to balance both poles simultaneously.

---

### 📂 Project Structure

```text
.
├── Dockerfile             # Docker image build instructions
├── docker-compose.yml     # Container orchestration for train/evaluate/plot
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template (for X11 display)
├── environment.py         # The custom OpenAI Gym environment (physics & rules)
├── train.py               # Script to train the PPO agent
├── evaluate.py            # Script to run, visualize, and record the agent
├── plot.py                # Script to generate learning curve graphs
├── README.md              # Project documentation
├── logs/                  # [Output] Directory for CSV training metrics
├── models/                # [Output] Directory for saved .zip models
└── media/                 # [Output] Directory for generated GIFs

```

---

### ⚙️ Environment Design (Inputs & Outputs)

The `DoublePendulumEnv` class simulates a cart on a horizontal track with two interconnected poles balancing on top.

* **Simulation Rate:** 60Hz ($dt = 1/60$).
* **Physics Engine:** `pymunk` (handles gravity, mass, inertia, and joint constraints).

**Input (Action Space):** * Type: 1D Continuous Box `[-1.0, 1.0]`

* Description: Represents the lateral force applied to the cart. This continuous value is scaled by a factor of 1000 before being applied to the physics body.

**Output (Observation Space):**

* Type: 6D Continuous Box
* Description: The state of the system returned to the agent every timestep. Contains normalized values:
`[cart_x, cart_vx, pole1_theta, pole1_omega, pole2_theta, pole2_omega]`

---

### 🧠 Reward Function Design

The reward function dictates how the agent learns. We implemented two distinct reward functions to demonstrate the importance of reward shaping.

**1. Baseline Reward**

* **Logic:** Purely prioritizes keeping the poles upright based on the cosine of their angles.
* **Equation:** 
$$R_{baseline} = \cos(\theta_1) + \cos(\theta_2)$$


* **Result:** The agent often learns erratic, unstable policies (e.g., violently shaking the cart to keep the poles up just long enough to maximize short-term reward).

**2. Shaped Reward**

* **Logic:** Augments the baseline goal with penalties to encourage stable, smooth, and energy-efficient control.
* **Equation:** 
$$R_{shaped} = R_{baseline} - 0.1|x_{cart}| - 0.01(|\omega_1| + |\omega_2|) - 0.001(a^2)$$


* **Breakdown:**
* **Center Penalty ($0.1|x_{cart}|$):** Keeps the cart near the center of the screen, preventing early termination from going out of bounds.
* **Velocity Penalty ($0.01(|\omega_1| + |\omega_2|)$):** Penalizes fast spinning, encouraging steady poles.
* **Action Penalty ($0.001(a^2)$):** Penalizes using excessive force, acting as an energy conservation constraint.



---

### 🚀 How to Run

This project is fully containerized. Ensure **Docker** and **Docker Compose** are installed and running.

#### Step 1: Build the Environment

Build the Docker image. This installs Python, `pymunk`, `stable-baselines3`, `PyTorch`, and necessary system dependencies for GUI rendering.

```bash
docker-compose build

```

#### Step 2: Train the Agents

We need to train two agents to compare their learning curves.

* **Inputs:** `--timesteps` (int), `--reward_type` (baseline/shaped), `--save_path` (string).
* **Outputs:** A `.zip` model file in `models/` and training metrics `.csv` in `logs/`.

Train the Baseline Agent:

```bash
docker-compose run train python train.py --timesteps 200000 --reward_type baseline --save_path models/baseline_model.zip

```

Train the Shaped Reward Agent:

```bash
docker-compose run train python train.py --timesteps 200000 --reward_type shaped --save_path models/shaped_model.zip

```

#### Step 3: Evaluate & Generate GIFs

Watch the trained agents attempt to balance the pendulum and save the results as GIFs.
*(Note: Windows/Mac users may need an X-server like VcXsrv/XQuartz running, and `.env` configured properly to see the live Pygame window).*

* **Inputs:** `--model_path` (string), `--gif_path` (string).
* **Outputs:** A live Pygame window and a generated `.gif` file in `media/`.

Generate Initial Agent GIF (Baseline model, erratic behavior):

```bash
docker-compose run evaluate python evaluate.py --model_path models/baseline_model.zip --gif_path media/agent_initial.gif

```

Generate Final Agent GIF (Shaped model, stable behavior):

```bash
docker-compose run evaluate python evaluate.py --model_path models/shaped_model.zip --gif_path media/agent_final.gif

```

#### Step 4: Generate Analysis Plot

Read the CSV files generated during training and plot the learning curves for comparison.

* **Inputs:** Reads from `logs/baseline` and `logs/shaped`.
* **Outputs:** Creates `reward_comparison.png` in the root directory.

```bash
docker-compose run train python plot.py

```
