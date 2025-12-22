from base_model import get_model
from args import get_args
from utils import model_detail

if __name__ == '__main__':
    args, ds_init = get_args()
    print(args)
    model = get_model(args)
    model_detail(model)

