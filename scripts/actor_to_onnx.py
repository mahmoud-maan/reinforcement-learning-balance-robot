import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import onnx
from stable_baselines3 import PPO

# 1. Define PyTorch wrapper to isolate the Actor
class ActorWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        # Extracts the shared/policy feature extractor and the action head
        self.action_net = policy.action_net
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Pass observation through policy features -> policy MLP -> action net
        features = self.features_extractor(observation)
        latent_pi, _ = self.mlp_extractor(features)
        action_logits_or_mean = self.action_net(latent_pi)
        
        # For Continuous Actions (Box), action_logits_or_mean is the deterministic mean action.
        # For Discrete Actions, uncomment the line below to get the action index:
        # return torch.argmax(action_logits_or_mean, dim=-1)
        
        return action_logits_or_mean

# 2. Parse arguments and load the trained SB3 PPO model
parser = argparse.ArgumentParser(description="Export SB3 PPO actor to ONNX")
parser.add_argument("model_path", help="Path to the trained .zip model file")
parser.add_argument("--model_name", default="ppo_actor", help="Name of the output ONNX model (default: ppo_actor)")
args = parser.parse_args()

model = PPO.load(args.model_path, device="cpu")

# 3. Instantiate and evaluate the actor wrapper
actor_model = ActorWrapper(model.policy)
actor_model.eval()

# 4. Create dummy input matching your observation space shape
obs_shape = (1,) + model.observation_space.shape
dummy_input = torch.randn(obs_shape, dtype=torch.float32)

# 5. Create models directory if it doesn't exist
models_dir = Path(__file__).parent.parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# 6. Export to ONNX
onnx_filename = models_dir / f"{args.model_name}.onnx"

torch.onnx.export(
    actor_model,
    dummy_input,
    str(onnx_filename),
    opset_version=18,
    export_params=True,
    dynamo=False,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

# 7. Ensure all weights are embedded into a single standalone file
onnx_model = onnx.load(str(onnx_filename))
onnx.save_model(onnx_model, str(onnx_filename), save_as_external_data=False)

# Clean up external data sidecar if one was generated during initial export
sidecar_file = str(onnx_filename) + ".data"
if os.path.exists(sidecar_file):
    os.remove(sidecar_file)

print(f"Successfully exported single-file policy actor to {onnx_filename}")