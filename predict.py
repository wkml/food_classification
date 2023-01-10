from PIL import Image
import os
import numpy as np
import json

import mindspore as ms
from mindspore import Tensor

from vit import ViT

def process_image(path):

    image_list = []

    mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
    std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])

    files = os.listdir(path)
    files = sorted(files, key=lambda x:int(x.split('.')[0]))
    
    for file in files:
        image = Image.open(path + file).convert("RGB")
        image = image.resize((224, 224))
        # plt.imshow(image)
        image = (image - mean) / std
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image_list.append(image)
    return image_list



if __name__ == "__main__":
    ms.set_context(device_target="CPU")
    images = process_image("./mindcon_xian/test/")

    network = ViT(num_classes=54)
    vit_path = "./ViT1/vit_b_16-10_104.ckpt"
    param_dict = ms.load_checkpoint(vit_path)
    ms.load_param_into_net(network, param_dict)

    model = ms.Model(network)

    results = []
    for image in images:
        pre = model.predict(Tensor(image,ms.float32))
        result = np.argmax(pre)
        results.append(result)
        print(result)
        
    with open("output.txt","w") as f:
        for result in results:
            f.write(str(result))
            f.write('\n')
    print(results)
    
    with open("./train_data/label_id_name.json") as f:
        result_dict = json.load(f)

    output_dict = [result_dict[str(i)] for i in results]
    with open("output_dict.txt","w") as f:
        for result in output_dict:
            f.write(str(result))
            f.write('\n')
    print(output_dict)
