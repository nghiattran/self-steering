import os
import random

if __name__ == '__main__':
    filepath = 'DATA/train/interpolated.csv'
    basepath = os.path.realpath(os.path.dirname(filepath))
    val_csv = os.path.join(basepath, 'val.csv')
    train_csv = os.path.join(basepath, 'train.csv')


    with open(filepath, 'r') as f:
        content = f.readlines()
    header = content[0]
    data = content[1:]
    random.shuffle(data)

    split = 500

    train_data = data[split:]
    val_data = data[:split]

    with open(train_csv, 'w') as f:
        f.write(header)
        f.write(''.join(train_data))

    with open(val_csv, 'w') as f:
        f.write(header)
        f.write(''.join(val_data))