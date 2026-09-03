import sys
import os
import numpy as np
import onnx
from onnx import numpy_helper

def parse_onnx_dense(onnx_path):
    model = onnx.load(onnx_path)
    graph = model.graph

    # Map initializers (weights & biases) by name
    weights_map = {}
    for init in graph.initializer:
        weights_map[init.name] = numpy_helper.to_array(init)

    layers = []
    
    # Process graph nodes in execution order
    for node in graph.node:
        if node.op_type in ["MatMul", "Gemm"]:
            # Extract weights from inputs
            w_name = [inp for inp in node.input if inp in weights_map][0]
            weight = weights_map[w_name]
            
            # Gemm handles transB and potential bias addition in one node
            bias = None
            if node.op_type == "Gemm":
                for attr in node.attribute:
                    if attr.name == "transB" and attr.i == 1:
                        weight = weight.T
                # Check for bias input
                if len(node.input) > 2 and node.input[2] in weights_map:
                    bias = weights_map[node.input[2]]

            layers.append({
                'type': 'Dense',
                'weight': weight,
                'bias': bias,
                'activation': 'Identity'
            })

        elif node.op_type == "Add":
            # If preceding layer lacks bias, attach this Add node as its bias
            b_name = [inp for inp in node.input if inp in weights_map]
            if b_name and layers and layers[-1]['bias'] is None:
                layers[-1]['bias'] = weights_map[b_name[0]]

        elif node.op_type in ["Relu", "Tanh", "Sigmoid"]:
            if layers:
                layers[-1]['activation'] = node.op_type

    return layers

def generate_c_header(layers, output_path):
    header_guard = os.path.basename(output_path).replace('.', '_').upper()
    header_guard = header_guard.replace('-', '_')
    
    in_dim = layers[0]['weight'].shape[0]
    out_dim = layers[-1]['weight'].shape[1]
    
    max_buf = in_dim
    for l in layers:
        max_buf = max(max_buf, l['weight'].shape[1])

    c_code = []
    c_code.append(f"#ifndef {header_guard}")
    c_code.append(f"#define {header_guard}\n")
    c_code.append("#include <math.h>\n")
    c_code.append(f"#define PPO_INPUT_DIM {in_dim}")
    c_code.append(f"#define PPO_OUTPUT_DIM {out_dim}")
    c_code.append(f"#define PPO_NUM_LAYERS {len(layers)}")
    c_code.append(f"")
    c_code.append(f"#define ACTOR_OBS_SIZE    PPO_INPUT_DIM")
    c_code.append(f"#define ACTOR_ACTION_SIZE PPO_OUTPUT_DIM\n")

    for i, layer in enumerate(layers):
        w = layer['weight']
        b = layer['bias'] if layer['bias'] is not None else np.zeros(w.shape[1], dtype=np.float32)
        
        rows, cols = w.shape
        c_code.append(f"// Layer {i}: Dense ({rows} -> {cols})")
        
        w_flat = ", ".join(f"{val:.8f}f" for val in w.flatten())
        c_code.append(f"static const float L{i}_WEIGHTS[{rows * cols}] = {{{w_flat}}};")
        
        b_flat = ", ".join(f"{val:.8f}f" for val in b.flatten())
        c_code.append(f"static const float L{i}_BIASES[{cols}] = {{{b_flat}}};\n")

    c_code.append(f"""\
// Performs forward inference without dynamic memory allocation
static inline void actor_forward(const float input[{in_dim}], float output[{out_dim}]) {{
    float buf_a[{max_buf}];
    float buf_b[{max_buf}];
    float* tmp; // Declared once at function level
    
    // Copy input into initial working buffer
    for (int i = 0; i < {in_dim}; i++) {{
        buf_a[i] = input[i];
    }}

    float* src = buf_a;
    float* dst = buf_b;
""")

    for i, layer in enumerate(layers):
        in_size = layer['weight'].shape[0]
        out_size = layer['weight'].shape[1]
        act = layer['activation']

        c_code.append(f"    // --- Layer {i} Execution ---")
        c_code.append(f"    for (int j = 0; j < {out_size}; j++) {{")
        c_code.append(f"        float sum = L{i}_BIASES[j];")
        c_code.append(f"        for (int k = 0; k < {in_size}; k++) {{")
        c_code.append(f"            sum += src[k] * L{i}_WEIGHTS[k * {out_size} + j];")
        c_code.append(f"        }}")

        if act == "Relu":
            c_code.append("        dst[j] = sum > 0.0f ? sum : 0.0f;")
        elif act == "Tanh":
            c_code.append("        dst[j] = tanhf(sum);")
        elif act == "Sigmoid":
            c_code.append("        dst[j] = 1.0f / (1.0f + expf(-sum));")
        else:
            c_code.append("        dst[j] = sum;")
            
        c_code.append("    }")

        # Swap double buffers (without re-declaring `tmp`)
        if i < len(layers) - 1:
            c_code.append("    tmp = src; src = dst; dst = tmp;\n")

    c_code.append(f"""
    // Copy final activation output to user buffer
    for (int i = 0; i < {out_dim}; i++) {{
        output[i] = dst[i];
    }}
}}

#endif // {header_guard}
""")

    with open(output_path, "w") as f:
        f.write("\n".join(c_code))

    print(f"Header generated successfully: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python onnx_actor_to_c.py <onnx_path> [output_path]")
        sys.exit(1)

    onnx_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = os.path.splitext(onnx_path)[0] + ".h"

    layers = parse_onnx_dense(onnx_path)
    generate_c_header(layers, output_path)

if __name__ == "__main__":
    main()