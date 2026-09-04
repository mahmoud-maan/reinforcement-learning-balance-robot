# Balancing Robot Sim-to-Real RL

This project explores how far a reinforcement learning policy trained in simulation can transfer to a real (Sim2Real) self-balancing robot. I used MuJoCo, Gymnasium, and Stable-Baselines3 to train PPO policies, then deployed them to an ESP32-based M5Stack Bala2 robot using custom Arduino firmware.
[BALA2 Fire Self-balancing Robot Kit](https://shop.m5stack.com/products/bala2-fire-self-balancing-robot-kit?srsltid=AfmBOorlHAX8QLWP_XKP2GJH_R4qbZQH9CwRpW_z0NvZVm21tJlj31ue&variant=43135408701697)

MuJoCo simulation video:

- [MuJoCo sim video](https://github.com/user-attachments/assets/db504bfa-e3d6-4a76-a719-b575fb7adb02)

## What I Did

- Studied the robot MJCF model and wrapped the MuJoCo robot into a custom Gymnasium environment.
- Defined the observation, action, reward, and termination logic needed for balancing.
- Trained a PPO policy with Stable-Baselines3 in simulation without domain randomization.
- Evaluated that policy in simulation and got promising balancing behavior.
- Exported the trained actor to ONNX and converted it into a C header for embedded inference.
- Wrote Arduino firmware to deploy the policy on the real ESP32-based robot.
- Compared simulation behavior against real-world behavior and used that gap to guide the next stage of training.
- Added a second training setup with domain randomization to improve robustness on hardware.

## Project Journey

### 1. Simulation Environment

I first focused on understanding the robot model and building RL environment around it.
That meant turning the MuJoCo MJCF model into a Gymnasium environment that Stable-Baselines3 could train on reliably.

### 2. First PPO Policy

I trained a PPO policy in MuJoCo without domain randomization.
In simulation, the robot balanced well and the results looked good enough to test on the real robot.

### 3. First Real Deployment

I deployed that first policy to the real M5Stack Bala2 running on the ESP32.
The robot could balance, which was a strong sign that sim-to-real transfer was working at least partially.

At the same time, the real robot exposed the limits of the first policy:

- It was not robust to pushes.
- It sometimes accelerated forward too aggressively and then fell.
- Small differences between simulation and hardware were enough to break stability.

Basic deployed policy video:

- [Basic policy real-robot video](https://github.com/user-attachments/assets/5450b334-f785-4316-a9a8-6f87825be672)

Sim-to-real showed that a policy trained in simulation can work on the real robot, but the real robot clearly exposed a gap between the simulator and reality. Domain randomization was my next step to make the policy more robust to that gap.

### 4. Domain Randomization

I then trained a domain-randomized policy to make the robot less sensitive to simulation mismatch.
The randomization included:

- Sensor noise
- Unknown friction
- Motor noise
- Tire ridges
- Action delay
- Battery discharge and voltage droop
- Unknown mass and density
- Random pushes

After deploying the domain-randomized policy, the robot behaved much better and looked noticeably more stable in the real world.

Domain-randomized policy video:

- [DR policy real-robot video](https://github.com/user-attachments/assets/187dcaec-91d2-4fa3-935a-5b3add36007b)

## Pipeline

1. Study the MuJoCo robot model and build a custom Gymnasium environment.
2. Train a PPO balancing policy in simulation with Stable-Baselines3.
3. Evaluate the learned policy in MuJoCo.
4. Export the actor network to ONNX.
5. Convert the ONNX actor into a self-contained C header.
6. Integrate the generated header into Arduino firmware.
7. Flash the firmware to the ESP32 robot and test on real hardware.

## Project Structure

```text
bala-rl/
├── envs/                  # custom Gymnasium environments
├── firmware/              # Arduino firmware for deployed policies
│   ├── balance_bot/       # basic policy firmware
│   ├── balance_bot_dr/    # domain-randomized policy firmware
│   └── imu_calibration/   # IMU calibration sketch
├── models/                # exported ONNX models
├── robot/                 # MJCF, URDF, CAD, and meshes
├── scripts/               # ONNX export and conversion utilities
├── train.py               # PPO training without DR
├── train_dr.py            # PPO training with DR
├── eval.py                # simulation evaluation for basic policy
└── eval_dr.py             # simulation evaluation for DR policy
```

## How To Run

Install dependencies:

```bash
uv sync
```

Train the basic policy:

```bash
uv run python train.py
```

Evaluate the basic policy in simulation:

```bash
uv run python eval.py --model runs/<run>/balance-bot-ppo_final.zip
```

Train the domain-randomized policy:

```bash
uv run python train_dr.py
```

Evaluate the domain-randomized policy:

```bash
uv run python eval_dr.py --model runs/<run>/balance-bot-ppo-dr_final.zip
```

Export a trained actor to ONNX:

```bash
uv run python scripts/actor_to_onnx.py runs/<run>/balance-bot-ppo_final.zip --model_name ppo_actor_basic
uv run python scripts/actor_to_onnx.py runs/<run>/balance-bot-ppo-dr_final.zip --model_name ppo_actor_dr
```

Convert ONNX to a C header:

```bash
uv run python scripts/onnx_actor_to_C.py models/ppo_actor_basic.onnx models/ppo_actor_basic.h
uv run python scripts/onnx_actor_to_C.py models/ppo_actor_dr.onnx models/ppo_actor_dr.h
```

## Inspiration

I learned a lot from Shawn Hymel's reinforcement learning series and blog posts, especially his balance robot work:

- [Shawn Hymel - Reinforcement Learning posts](https://shawnhymel.com/tag/reinforcement-learning/)
- [Shawn Hymel - Reinforcement Learning for a Balance Robot](https://shawnhymel.com/3219/an-idea-im-exploring-reinforcement-learning-for-a-balance-robot/)

## Next Step

I am currently working on adding command inputs to control the robot and exploring reinforcement learning with goal-conditioned policies.
