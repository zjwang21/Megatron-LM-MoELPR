import argparse
from transformers import MixtralConfig, MixtralForCausalLM, AutoTokenizer, AutoModelForCausalLM, Qwen2MoeConfig, Qwen2MoeForCausalLM
import torch
import os
import math
import json

def split_params(args, params, name, config, index):
    if 'up' in name or 'gate' in name:
        assert params.shape == (config.intermediate_size, config.hidden_size)
        return params[index, :]
    elif 'down' in name:
        assert params.shape == (config.hidden_size, config.intermediate_size)
        return params[:, index]
    else:
        raise NotImplementedError

def kmeans_cluster():
    pass

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Receive upcycling model's args")
    parser.add_argument("--model_path", default='meta-llama/Llama-2-7b-hf', type=str, help="original model path")
    parser.add_argument("--output_path", default=None, type=str, help="upcycled model ckpt save path")
    parser.add_argument("--num_experts", default=4, type=int, help="upcycled model num experts")
    parser.add_argument("--moe_type", default="sparse", type=str, help="upcycled model type")
    parser.add_argument("--moe_ffn_dim", default=0.25, type=float, help="upcycled model type")

    # Parse the arguments
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.save_pretrained(args.output_path)
    
    if args.moe_type == "sparse":
        config = MixtralConfig.from_pretrained(args.model_path)
        setattr(config, 'model_type', 'mixtral')
        setattr(config, 'architectures', ["MixtralForCausalLM"])
        setattr(config, 'num_local_experts', args.num_experts)
        config.save_pretrained(args.output_path)
    elif args.moe_type == "share":
        config = Qwen2MoeConfig.from_pretrained(args.model_path)
        setattr(config, 'model_type', 'qwen2_moe')
        setattr(config, 'architectures', ["Qwen2MoeForCausalLM"])
        setattr(config, 'shared_expert_intermediate_size', config.intermediate_size)
        setattr(config, 'moe_intermediate_size', math.ceil(config.intermediate_size * args.moe_ffn_dim))
        setattr(config, 'num_experts', args.num_experts)
        config.save_pretrained(args.output_path)
    else:
        raise NotImplementedError

    print(config)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    ckpt = model.state_dict()
    layers = len(model.model.layers)

    output = {k:v for k,v in ckpt.items() if 'mlp' not in k}

    if args.moe_type == "sparse":
        for i in range(layers):
            for k in ckpt:
                if 'mlp' in k and ('layers.' + str(i) + '.') in k:
                    for j in range(args.num_experts):
                        if 'mlp.gate_proj' in k:
                            output[k.replace('mlp.gate_proj', 'block_sparse_moe.experts.' + str(j) + '.w1')] = ckpt[k]
                        if 'mlp.up_proj' in k:
                            output[k.replace('mlp.up_proj', 'block_sparse_moe.experts.' + str(j) + '.w3')] = ckpt[k]
                        if 'mlp.down_proj' in k:
                            output[k.replace('mlp.down_proj', 'block_sparse_moe.experts.' + str(j) + '.w2')] = ckpt[k]
    elif args.moe_type == "share":
        for i in range(layers):
            index = torch.randint(0, config.intermediate_size, (config.moe_intermediate_size,))
            for k in ckpt:
                if 'mlp' in k and ('layers.' + str(i) + '.') in k:
                    output[k.replace('mlp.', 'mlp.shared_expert.')] = ckpt[k]
                    for j in range(args.num_experts):
                        output[k.replace('mlp.', 'mlp.experts.' + str(j) + '.')] = split_params(args, ckpt[k], k, config, index)
    else:
        raise NotImplementedError

    torch.save(output, os.path.join(args.output_path, "pytorch_model.bin"))

if __name__ == "__main__":
    main()

