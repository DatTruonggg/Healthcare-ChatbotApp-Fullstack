import torch
from typing import Mapping, Any
import functools
import orbax.checkpoint


Params = Mapping[str, Any]

def load_and_format_params(path: str) -> dict:
    params = load_params(path)
    param_state = torch.nn.ParameterDict()
    for key, value in params.items():
        param_state[key] = torch.as_tensor(value)
    # Nếu params không phải là một từ điển, bạn có thể sử dụng các hàm tương ứng như torch.as_tensor() hoặc torch.Tensor()
    # param_state = torch.as_tensor(params)
    # param_state = torch.Tensor(params)
    # Tiếp tục xử lý các tham số nếu cần thiết
    remapped_params = param_remapper(param_state)
    nested_params = nest_params(remapped_params)
    return nested_params


@functools.cache
def load_params(path: str) -> Params:
    """Loads parameters from a checkpoint path."""
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
    nested_params = {}
    for path, param in params.items():
        *path, leaf = path.split('/')
        subdict = nested_params
        for key in path:
            subdict = subdict.setdefault(key, {})
        subdict[leaf] = param
    return nested_params