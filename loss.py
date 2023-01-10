from mindspore.nn import LossBase
from mindspore import nn, ops
import mindspore as ms
from mindspore.common.initializer import One

class CrossEntropySmooth(LossBase):
    """CrossEntropy."""

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = ms.Tensor(1.0 - smooth_factor, ms.float32)
        self.off_value = ms.Tensor(1.0 * smooth_factor / (num_classes - 1), ms.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss

if __name__ == '__main__' :
    logits = ms.Tensor(shape = (32, 54), dtype=ms.float32, init=One())
    label = ms.Tensor(shape = (32,), dtype=ms.int32, init=One())
    network_loss = CrossEntropySmooth(sparse=True,
                                  reduction="mean",
                                  smooth_factor=0.1,
                                  num_classes=54)
    loss = network_loss(logits, label)
    print(loss)