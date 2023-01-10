from dataset import get_dataset
from vit import ViT
import mindspore as ms
from mindspore import nn

if "__main__" == __name__:
    ms.set_context(mode=ms.GRAPH_MODE,device_target="Ascend")
    vit = ViT()
    vit.dense = nn.Dense(768, 54)
    train_data, val_data = get_dataset()
    for data, label in train_data:
        print(vit(data).shape)
        print(label.shape)
        break