from .armodel import ARModel
from .cnn1d import CNN1d
from .cnn2d import resnet

model_dict = {
    "cnn1d": CNN1d,
    "ARConvs": ARModel,
    "cnn2d": resnet,
}


def create_model(model_name: str, model_hparams: dict):
    return model_dict[model_name](**model_hparams)
