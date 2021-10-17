import mindspore as ms
from mindspore.common import Parameter
import participant_model

net = participant_model.Net()

bamboo_dict = ms.load_checkpoint('./bamboo_conv.ckpt')
rcnn_dict = ms.load_checkpoint('./faster_rcnn-12_1000.ckpt')
fusion_dict = {'bamboo_net': {}, 'rcnn_net': {}}
for k, v in bamboo_dict.items():
    fusion_dict['bamboo_net'][k] = Parameter(v, k)
for k, v in rcnn_dict.items():
    fusion_dict['rcnn_net'][k] = Parameter(v, k)

ms.save_checkpoint(net, 'bamboo_rcnn.ckpt')
