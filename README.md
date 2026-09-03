# Balance Bot - Sim-to-Real RL

This project is my attempt to train a self-balancing robot in simulation and run that policy on real hardware.
I used reinforcement learning to control a 2-wheel robot and deployed the result to an ESP32-based M5Stack Bala2.

## What I've done

- Custom MuJoCo + Gymnasium environments for balancing.
- Trained PPO policies in different setups, including domain randomization.
- Exported trained actors to ONNX and converted them into C headers for embedded use.
- Wrote Arduino firmware to run inference on-device and control motors in real time.

## Tech Used

- Python, Stable-Baselines3, Gymnasium, MuJoCo
- ONNX + custom ONNX-to-C conversion
- Arduino C++ on ESP32 (M5Stack Bala2)

## Status

Work in progress. I will add control commands also I am actively tuning firmware behavior and improving real-world stability.
