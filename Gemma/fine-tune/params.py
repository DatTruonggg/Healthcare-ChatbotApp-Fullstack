import functools
from typing import Any, Mapping
import torch
import orbax.checkpoint

Params = Mapping[str, Any]


import torch

def load_and_format_params(path: str) -> dict:
    params = load_params(path)
    param_state = torch.tree_map(torch.tensor, params)
    remapped_params = param_remapper(param_state)
    nested_params = nest_params(remapped_params) 
    return nested_params



@functools.cache
def load_params(path: str) -> Params:
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    params = checkpointer.restore(path)
    return params


def param_remapper(orig_params: Params) -> Params:
    new_params = {}
    for k, v in orig_params.items():
        if 'mlp/' in k:
            layer_name, param = k.rsplit('/', maxsplit=1)
        if layer_name not in new_params:
            new_params[layer_name] = {}
        if 'w' in v:
            new_params[layer_name][param] = v['w']
        else:
            new_params[k] = v
    return new_params


def nest_params(params: Params) -> Params:
    """Nests params as a dict of dicts rather than a flat dict."""
    nested_params = {}
    for path, param in params.items():
        *path, leaf = path.split('/')
        subdict = nested_params
        for key in path:
            subdict = subdict.setdefault(key, {})
        subdict[leaf] = param
    return nested_params