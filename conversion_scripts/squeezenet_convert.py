import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer
import torchvision as tv
from torchvision.transforms import v2
# from lpips.pretrained_networks import squeezenet
from IPython import embed
import torch

from torchvision import models as tvm

device='cpu'

model = tvm.squeezenet1_1().eval().to(device)

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]).to(device)

ex_ref = transforms(tv.io.read_image('../imgs/ex_ref.png'))
input_batch = ex_ref.unsqueeze(0).to(device)

model(input_batch)


keras_model = nobuco.pytorch_to_keras(
    model,
    args=[input_batch], kwargs=None,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    trace_shape=True,
)

keras_model.save('squeezenet' + '.keras')

print(keras_model.summary())