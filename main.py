import mindspore as ms
from vit import ViT
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore import nn
from loss import CrossEntropySmooth
from dataset import get_dataset

class CustomWithLossCell(nn.Cell):
    """连接前向网络和损失函数"""

    def __init__(self, backbone, loss_fn):
        """前向网络backbone和损失函数loss_fn"""
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label):
        output = self._backbone(data)                 # 前向计算得到网络输出
        return self._loss_fn(output, label)  # 得到多标签损失值

class CustomWithEvalCell(nn.Cell):
    """自定义多标签评估网络"""

    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label):
        output = self.network(data)
        return output, label

if __name__ == '__main__' :
    ms.set_context(mode=ms.GRAPH_MODE,device_target="Ascend")
    train_data, val_data = get_dataset()

    epoch_size = 5
    momentum = 0.9
    num_classes = 10
    step_size = train_data.get_dataset_size()

    network = ViT()
    vit_path = "./ckpt/vit_b_16_224.ckpt"
    param_dict = ms.load_checkpoint(vit_path)
    ms.load_param_into_net(network, param_dict)
    network.dense = nn.Dense(768, num_classes)

    lr = nn.cosine_decay_lr(min_lr=float(0),
                        max_lr=0.00005,
                        total_step=epoch_size * step_size,
                        step_per_epoch=step_size,
                        decay_epoch=10)
    network_opt = nn.Adam(network.trainable_params(), lr, momentum)
    network_loss = CrossEntropySmooth(sparse=True,
                                    reduction="mean",
                                    smooth_factor=0.0,
                                    num_classes=num_classes)

    loss_net = CustomWithLossCell(network, network_loss)
    eval_net = CustomWithEvalCell(network)
    
    ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=100)
    ckpt_callback = ModelCheckpoint(prefix='vit_b_16', directory='./ViT_food2', config=ckpt_config)
    ascend_target = (ms.get_context("device_target") == "Ascend")
    if ascend_target:
        model = ms.Model(loss_net, optimizer=network_opt, eval_network=eval_net, metrics={"acc"}, amp_level="O2")
    else:
        model = ms.Model(loss_net, optimizer=network_opt, eval_network=eval_net, metrics={"acc"}, amp_level="O0")
    model.train(epoch_size,
            train_data,
            callbacks=[ckpt_callback, LossMonitor(step_size), TimeMonitor(step_size)]
            )
    result = model.eval(val_data)
    print(result)