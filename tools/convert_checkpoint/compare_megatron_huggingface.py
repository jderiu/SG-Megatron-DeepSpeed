import sys
import torch
import os
from collections import OrderedDict
from pathlib import Path
from transformers.models.bloom.modeling_bloom import BloomModel


# insert megatron's root dir into sys.path
root_repo_path = str(Path(__file__).resolve().parents[2])
if root_repo_path not in sys.path:
    sys.path.insert(0, root_repo_path)


def compare_data(datum0, datum1, name_list=[]):
    assert type(datum0) == type(datum1), f'type mismatch: {type(datum0)} != {type(datum1)}'

    if type(datum0) in (dict, OrderedDict):
        assert len(datum0) == len(datum1), f'length mismatch: {len(datum0)} != {len(datum1)}'
        for (k0, v0), (k1, v1) in zip(datum0.items(),datum1.items()):
            compare_data(v0, v1, name_list + [str(k0)])
    elif type(datum0) in (list, tuple):
        for v0, v1 in zip(datum0, datum1):
            compare_data(v0, v1, name_list)
    elif torch.is_tensor(datum0):
        assert datum0.shape == datum1.shape, f'shape mismatch: {datum0.shape} != {datum1.shape}'
        diff_max = (datum0 - datum1).abs().max()
        diff_sum = (datum0 - datum1).abs().sum()
        diff_mean = (datum0 - datum1).abs().mean()
        prefix = '.'.join(name_list)
        print(f'[diff] {prefix} = {diff_max}, {diff_sum}, {diff_mean}')
    else:
        #pass
        prefix = '.'.join(name_list)
        print(f'[other] {prefix} = {datum0}')


def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <checkpoint file>')
        exit(1)

    ckpt_file = sys.argv[1]
    if not os.path.isfile(ckpt_file):
        print(f'{ckpt_file} is not a valid file')
        sd0 = BloomModel.from_pretrained(ckpt_file).to(torch.float16).state_dict()
    else:
        print(f'loading checkpoint file: {ckpt_file}')
        sd0 = torch.load(ckpt_file, map_location=torch.device('cpu'))

    ckpt_file = sys.argv[2]
    if not os.path.isfile(ckpt_file):
        print(f'{ckpt_file} is not a valid file')
        exit(1)

    print(f'loading checkpoint file: {ckpt_file}')
    sd1 = torch.load(ckpt_file, map_location=torch.device('cpu'))['model']['language_model']
    new_dict = OrderedDict()
    new_dict['word_embeddings.weight'] = sd1['embedding']['word_embeddings']['weight']
    new_dict['word_embeddings_ln.weight'] = sd1['embedding']['word_embeddings_layernorm']['weight']
    new_dict['word_embeddings_ln.bias'] = sd1['embedding']['word_embeddings_layernorm']['bias']
    sd1 = OrderedDict(list(new_dict.items()) + list(sd1['encoder'].items()))




    compare_data(sd0, sd1)

    quit()


if __name__ == "__main__":
    main()
