import argparse
import os.path

import torch
from transformers import BloomConfig
from transformers.models.bloom.convert_bloom_original_checkpoint_to_pytorch import convert_bloom_checkpoint_to_pytorch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--bloom_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the Megatron-LM checkpoint path.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--bloom_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--shard_model",
        action="store_true",
        help="An optional setting to shard the output model \nThis enables sharding the converted checkpoint",
    )
    parser.add_argument(
        "--pretraining_tp",
        default=4,
        type=int,
        help="Pretraining TP rank that has been used when training the model in Megatron-LM \n",
    )
    parser.add_argument(
        "--bloom_model",
        default='bigscience/bloom-1b1',
        type=str,
        help="Which Size\n",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pytorch_dump_folder_path):
        os.makedirs(args.pytorch_dump_folder_path)

    config = BloomConfig.from_pretrained(args.bloom_model)
    config.torch_dtype = torch.float16
    config.to_json_file(os.path.join(args.pytorch_dump_folder_path, 'config.json'))


    convert_bloom_checkpoint_to_pytorch(
        args.bloom_checkpoint_path,
        os.path.join(args.pytorch_dump_folder_path, 'config.json'),
        args.pytorch_dump_folder_path,
        args.shard_model,
        args.pretraining_tp,
    )
