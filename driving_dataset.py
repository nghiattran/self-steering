import os
import random

if __name__ == '__main__':
    filepath = 'DATA/driving_dataset/data.txt'
    basepath = os.path.realpath(os.path.dirname(filepath))
    val_csv = os.path.join(basepath, 'val.csv')
    train_csv = os.path.join(basepath, 'train.csv')
    header = 'filename,angle\n'
    with open(filepath, 'r') as f:
        data = f.readlines()

    # for i in range(len(data)):
    #     entry = data[i]
    #     file, angle = entry.split(' ')
    #     file = os.path.join(basepath, file)
    #     data[i] = ','.join([file, angle])

    random.shuffle(data)

    split = 500

    train_data = data[split:]
    val_data = data[:split]

    with open(train_csv, 'w') as f:
        # f.write(header)
        f.write(''.join(train_data))

    with open(val_csv, 'w') as f:
        # f.write(header)
        f.write(''.join(val_data))