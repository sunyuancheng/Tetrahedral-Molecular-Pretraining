import torch
import torch.distributed as dist
from torch.hub import load_state_dict_from_url


def load_pretrained_model(model, pretrained_model_path):
    loaded_state_dict = torch.load(str(pretrained_model_path))['model']
    model_state_dict = model.state_dict()
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        new_param_name = param_name
        # if new_param_name not in model_state_dict:
        #     print(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        # elif model_state_dict[new_param_name].shape != loaded_state_dict[param_name].shape:
        #     print(f'Pretrained parameter "{param_name}" '
        #             f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
        #             f'model parameter of shape {model_state_dict[new_param_name].shape}.')
        # else:
        #     print(f'Loading pretrained parameter "{param_name}".')
        #     pretrained_state_dict[new_param_name] = loaded_state_dict[param_name]
        
        if new_param_name in model_state_dict and model_state_dict[new_param_name].shape == loaded_state_dict[param_name].shape:
            pretrained_state_dict[new_param_name] = loaded_state_dict[param_name]
        model_state_dict.update(pretrained_state_dict)

    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Successfully load checkpoint from {pretrained_model_path}!")
        dist.barrier()
        return model_state_dict
    else:
        print(f"Successfully load checkpoint from {pretrained_model_path}!")
    return model_state_dict
