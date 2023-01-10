import os
import shutil

if __name__ == '__main__':
    path = './train_data/train_data/'

    target_path = './data/train/'

    files = os.listdir(path)
    for file in files:
        if file.endswith('.txt'):
            with open(path + file,'r') as f:
                line = f.read()
                image, label = line.split(', ')
                if not os.path.exists(target_path + label):
                    os.makedirs(target_path + label)
                shutil.copyfile(path + image, target_path + label + '/' + image)