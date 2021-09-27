import glob
import os
import shutil
import argparse as ap
from random import shuffle


def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--size', default=0.7)

    return parser.parse_args()


def main():
    args = get_args()

    dir_content = glob.glob(args.dir + '/*.*')
    shuffle(dir_content)
    train_size = int(len(dir_content) * args.size)

    train_path = os.path.join(args.dir, 'train')
    test_path = os.path.join(args.dir, 'test')
    os.mkdir(train_path)
    os.mkdir(test_path)

    counter = 0

    while counter < len(dir_content):
        src = dir_content[counter]
        file_name = os.path.basename(src)
        if counter <= train_size:
            dst = os.path.join(args.dir, 'train', file_name)
            print(f'{file_name} -> train')
            shutil.move(src, dst)
        else:
            dst = os.path.join(args.dir, 'test', file_name)
            print(f'{file_name} -> test')
            shutil.move(src, dst)
        counter += 1

    print(f'Moved {len(os.listdir(train_path))} files to train dir')
    print(f'Moved {len(os.listdir(test_path))} files to test dir')


if __name__ == '__main__':
    main()
