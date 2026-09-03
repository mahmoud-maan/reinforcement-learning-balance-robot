import argparse
import os
import tempfile
from pathlib import Path
import subprocess
import numpy as np
import onnxruntime as ort
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser(description="Verify SB3, ONNX, and C outputs match")
    parser.add_argument("model_path", help="Path to the trained .zip model file")
    args = parser.parse_args()

    model_path = args.model_path
    onnx_path = str(Path(__file__).parent.parent / "models" / "ppo_actor.onnx")
    c_header_path = str(Path(__file__).parent.parent / "models" / "ppo_actor.h")

    # Define test observation input
    # obs: [-0.00092925, 0.04433497, 0.00306454, 0.00351217] expected action: [0.00656078 0.00591131]
    sample_obs = [-0.00092925, 0.04433497, 0.00306454, 0.00351217]
    test_obs = np.array(sample_obs, dtype=np.float32)
    # -------------------------------------------------------------
    # 1. SB3 PPO Model Prediction
    # -------------------------------------------------------------
    sb3_model = PPO.load(model_path, device="cpu")
    sb3_action, _ = sb3_model.predict(test_obs, deterministic=True)
    sb3_action = np.atleast_1d(sb3_action).flatten()

    # -------------------------------------------------------------
    # 2. ONNX Model Direct Inference
    # -------------------------------------------------------------
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    onnx_action = ort_session.run(None, {input_name: test_obs.reshape(1, -1)})[0].flatten()

    # -------------------------------------------------------------
    # 3. Existing C Header Inference
    # -------------------------------------------------------------
    in_dim = len(test_obs)
    out_dim = len(sb3_action)
    obs_str = ", ".join(f"{x:.8f}f" for x in test_obs)

    c_runner = f"""
#include <stdio.h>
#include "{c_header_path}"

int main() {{
    float observation[{in_dim}] = {{{obs_str}}};
    float action[{out_dim}];

    ppo_actor_predict(observation, action);

    for (int i = 0; i < {out_dim}; i++) {{
        printf("%.8f%s", action[i], (i == {out_dim} - 1) ? "" : ", ");
    }}
    printf("\\n");
    return 0;
}}
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner_c = os.path.join(tmp_dir, "runner.c")
        runner_bin = os.path.join(tmp_dir, "runner")

        with open(runner_c, "w") as f:
            f.write(c_runner)

        # Compile and execute C runner
        subprocess.run(["gcc", "-O2", runner_c, "-o", runner_bin, "-lm"], check=True)
        c_output_str = subprocess.run([runner_bin], capture_output=True, text=True, check=True).stdout
    c_action = np.array([float(x) for x in c_output_str.strip().split(",")], dtype=np.float32)

    # -------------------------------------------------------------
    # Output Comparison
    # -------------------------------------------------------------
    print("\n==========================================")
    print("      MODEL PREDICTION COMPARISON")
    print("==========================================")
    print(f"Observation:     {test_obs.tolist()}\n")
    print(f"1. SB3 Action:   {sb3_action.tolist()}")
    print(f"2. ONNX Action:  {onnx_action.tolist()}")
    print(f"3. C Action:     {c_action.tolist()}")
    print("------------------------------------------")

    if np.allclose(sb3_action, onnx_action, rtol=1e-4) and np.allclose(onnx_action, c_action, rtol=1e-4):
        print("✅ SUCCESS: All 3 outputs match perfectly!")
    else:
        print("❌ ERROR: Mismatch detected between models.")


if __name__ == "__main__":
    main()