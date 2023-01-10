import mindspore as ms
from vit import ViT
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore import nn, context
from loss import CrossEntropySmooth
from dataset import get_dataset

if __name__ == '__main__' :
    ms.set_context(mode=ms.GRAPH_MODE,device_target="Ascend")
    train_data, val_data = get_dataset()

    network = ViT(num_classes=54)
    vit_path = "./ViT1/vit_b_16-10_104.ckpt"
    param_dict = ms.load_checkpoint(vit_path)
    ms.load_param_into_net(network, param_dict)

    epoch_size = 10
    momentum = 0.9
    num_classes = 54
    step_size = train_data.get_dataset_size()

    lr = nn.cosine_decay_lr(min_lr=float(0),
                        max_lr=0.00005,
                        total_step=epoch_size * step_size,
                        step_per_epoch=step_size,
                        decay_epoch=10)
    network_opt = nn.Adam(network.trainable_params(), lr, momentum)
    network_loss = CrossEntropySmooth(sparse=True,
                                    reduction="mean",
                                    smooth_factor=0.1,
                                    num_classes=num_classes)

    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}
    ascend_target = (ms.get_context("device_target") == "Ascend")
    if ascend_target:
        model = ms.Model(network, network_loss, optimizer=network_opt, metrics=eval_metrics, amp_level="O2")
    else:
        model = ms.Model(network, network_loss, optimizer=network_opt, metrics=eval_metrics, amp_level="O0")

    result = model.eval(val_data)
    print(result)