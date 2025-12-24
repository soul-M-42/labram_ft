from base_model import get_model
from args_yaml import get_args_from_yaml
from utils import model_detail

if __name__ == '__main__':
    args, ds_init = get_args_from_yaml('arg_ft.yaml')
    print(args)
    model = get_model(args)
    model_detail(model)

