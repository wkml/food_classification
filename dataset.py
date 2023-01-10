import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision

def get_dataset():
    image_folder_dataset_dir = "./data/train"
    mapping = {}
    for i in range(54):
        mapping[str(i)] = int(i)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    transforms_list = [vision.RandomCropDecodeResize(size=224,
                                                    scale=(0.08, 1.0),
                                                    ratio=(0.75, 1.333)),
                    vision.RandomHorizontalFlip(0.75),
                    vision.Normalize(mean=mean, std=std),
                    vision.HWC2CHW(),
                    ]

    dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
                                    shuffle=True,
                                    num_parallel_workers=8,
                                    class_indexing=mapping
                                    )

    dataset = dataset.map(operations=transforms_list)

    dataset = dataset.batch(32, drop_remainder=True, num_parallel_workers=8)

    train_dataset, val_dataset = dataset.split([0.9, 0.1])
    
    return train_dataset, val_dataset

if "__main__" == __name__:
    train_data, val_data = get_dataset()
    for data, label in train_data:
        print(type(data))
        break