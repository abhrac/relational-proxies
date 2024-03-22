import os
import csv
import numpy as np


def organise_cub_dataset(data_path):

    split = {}
    train_count = {}
    test_count = {}

    os.system('mkdir {}/train'.format(data_path))
    os.system('mkdir {}/test'.format(data_path))

    with open(os.path.join(data_path, 'classes.txt')) as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            class_id, class_name = line.split(' ')
            folder_name = class_name.split('.')[0]
            train_count[folder_name] = 1
            test_count[folder_name] = 1
            os.system('mkdir {}/train/{}'.format(data_path, 'class_' + str(folder_name)))
            os.system('mkdir {}/test/{}'.format(data_path, 'class_' + str(folder_name)))
            line = fp.readline()

    with open(os.path.join(data_path, 'train_test_split.txt')) as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            image_id, image_split = line.split(' ')
            split[int(image_id)] = 'train' if int(image_split) else 'test'
            line = fp.readline()

    with open(os.path.join(data_path, 'images.txt')) as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            image_id, image_path = line.split(' ')
            img_class = image_path.split('.')[0]
            image_split = split[int(image_id)]
            full_image_path = r'{}/images/{}'.format(data_path, image_path)
            if image_split == 'train':
                new_image_path = os.path.join(data_path, 'train', 'class_' + img_class, str(int(image_id)).zfill(5) + '.jpg')
                print('{} {}'.format(full_image_path, new_image_path))
                os.system('cp {} {}'.format(full_image_path, new_image_path))
            else:
                new_image_path = os.path.join(data_path, 'test', 'class_' + img_class, str(int(image_id)).zfill(5) + '.jpg')
                print('{} {}'.format(full_image_path, new_image_path))
                os.system('cp {} {}'.format(full_image_path, new_image_path))
            line = fp.readline()


def organise_nabirds_dataset(data_path):

    split = {}
    class_dict = {}

    os.system('mkdir {}/train'.format(data_path))
    os.system('mkdir {}/test'.format(data_path))

    classes = []

    with open(os.path.join(data_path, 'images.txt')) as fp:
        lines = fp.readlines()
        lines = sorted([line.strip() for line in lines])
        for i in range(len(lines)):
            line = lines[i]
            image_id, image_path = line.split(' ', maxsplit=1)
            class_id, image_nm = image_path.split('/', maxsplit=1)
            classes.append(int(class_id))
    classes = np.unique(classes)
    for i in range(len(classes)):
        class_dict[str(classes[i]).zfill(4)] = i + 1

    for i in range(len(classes)):
        os.system('mkdir {}/train/{}'.format(data_path, 'class_' + str(i + 1).zfill(3)))
        os.system('mkdir {}/test/{}'.format(data_path, 'class_' + str(i + 1).zfill(3)))

    with open(os.path.join(data_path, 'train_test_split.txt')) as fp:
        lines = fp.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            image_id, image_split = line.split(' ', maxsplit=1)
            split[image_id] = 'train' if int(image_split) else 'test'

    with open(os.path.join(data_path, 'images.txt')) as fp:
        lines = fp.readlines()
        lines = sorted([x.strip() for x in lines])
        for line in lines:
            image_id, image_path = line.split(' ', maxsplit=1)
            img_class, image_nm = image_path.split('/', maxsplit=1)
            image_split = split[image_id]
            full_image_path = r'{}/images/{}'.format(data_path, image_path)
            img_class_base_0 = str(class_dict[img_class]).zfill(3)
            if image_split == 'train':
                new_image_path = os.path.join(data_path, 'train', 'class_' + img_class_base_0, image_nm)
                print('{} {}'.format(full_image_path, new_image_path))
                os.system('cp {} {}'.format(full_image_path, new_image_path))
            else:
                new_image_path = os.path.join(data_path, 'test', 'class_' + img_class_base_0, image_nm)
                print('{} {}'.format(full_image_path, new_image_path))
                os.system('cp {} {}'.format(full_image_path, new_image_path))


def organise_stanford_cars_dataset(data_path):

    os.system('mkdir {}/train'.format(data_path))
    os.system('mkdir {}/test'.format(data_path))

    with open(os.path.join(data_path, 'devkit', 'anno_train.csv'), 'r') as f:
        reader = csv.reader(f)
        train_data = np.array(list(reader))
    with open(os.path.join(data_path, 'devkit', 'anno_test.csv'), 'r') as f:
        reader = csv.reader(f)
        test_data = np.array(list(reader))

    max_class = train_data[:, 5].astype(int).max()

    for i in range(max_class):
        os.system('mkdir {}/train/{}'.format(data_path, 'class_' + str(i + 1).zfill(3)))
        os.system('mkdir {}/test/{}'.format(data_path, 'class_' + str(i + 1).zfill(3)))

    for i in range(train_data.shape[0]):
        full_image_path = os.path.join(data_path, 'cars_train', train_data[i][0])
        # print(full_image_path)
        os.system('cp {} {}/train/class_{}/{}'.format(full_image_path, data_path, str(train_data[i][5]).zfill(3), train_data[i][0]))

    for i in range(test_data.shape[0]):
        full_image_path = os.path.join(data_path, 'cars_test', test_data[i][0])
        # print(full_image_path)
        os.system('cp {} {}/test/class_{}/{}'.format(full_image_path, data_path, str(test_data[i][5]).zfill(3), test_data[i][0]))


def organise_fgvc_aircraft_dataset(data_path):

    os.system('mkdir {}/train'.format(data_path))
    os.system('mkdir {}/val'.format(data_path))
    os.system('mkdir {}/trainval'.format(data_path))
    os.system('mkdir {}/test'.format(data_path))

    class_dict = {}

    with open(os.path.join(data_path, 'data', 'variants.txt')) as f:
        content = f.readlines()
        classes = sorted([x.strip() for x in content])
        for i in range(len(classes)):
            class_dict[classes[i]] = i + 1

    for i in range(len(classes)):
        os.system('mkdir {}/train/{}'.format(data_path, 'class_' + str(i + 1).zfill(3)))
        os.system('mkdir {}/val/{}'.format(data_path, 'class_' + str(i + 1).zfill(3)))
        os.system('mkdir {}/trainval/{}'.format(data_path, 'class_' + str(i + 1).zfill(3)))
        os.system('mkdir {}/test/{}'.format(data_path, 'class_' + str(i + 1).zfill(3)))

    with open(os.path.join(data_path, 'data', 'images_variant_train.txt')) as f:
        train_content = f.readlines()
        train_content = [x.strip() for x in train_content]
        for tc in train_content:
            image_id, class_name = tc.split(' ', maxsplit=1)
            class_id = class_dict[class_name]
            full_image_path = os.path.join(data_path, 'data', 'images', image_id + '.jpg')
            new_image_path = os.path.join(data_path, 'train', 'class_' + str(class_id).zfill(3), image_id + '.jpg')
            print('{} {}'.format(full_image_path, new_image_path))
            os.system('cp {} {}'.format(full_image_path, new_image_path))

    with open(os.path.join(data_path, 'data', 'images_variant_val.txt')) as f:
        val_content = f.readlines()
        val_content = [x.strip() for x in val_content]
        for vc in val_content:
            image_id, class_name = vc.split(' ', maxsplit=1)
            class_id = class_dict[class_name]
            full_image_path = os.path.join(data_path, 'data', 'images', image_id + '.jpg')
            new_image_path = os.path.join(data_path, 'val', 'class_' + str(class_id).zfill(3), image_id + '.jpg')
            print('{} {}'.format(full_image_path, new_image_path))
            os.system('cp {} {}'.format(full_image_path, new_image_path))

    with open(os.path.join(data_path, 'data', 'images_variant_trainval.txt')) as f:
        trainval_content = f.readlines()
        trainval_content = [x.strip() for x in trainval_content]
        for tvc in trainval_content:
            image_id, class_name = tvc.split(' ', maxsplit=1)
            class_id = class_dict[class_name]
            full_image_path = os.path.join(data_path, 'data', 'images', image_id + '.jpg')
            new_image_path = os.path.join(data_path, 'trainval', 'class_' + str(class_id).zfill(3), image_id + '.jpg')
            print('{} {}'.format(full_image_path, new_image_path))
            os.system('cp {} {}'.format(full_image_path, new_image_path))

    with open(os.path.join(data_path, 'data', 'images_variant_test.txt')) as f:
        test_content = f.readlines()
        test_content = [x.strip() for x in test_content]
        for tc in test_content:
            image_id, class_name = tc.split(' ', maxsplit=1)
            class_id = class_dict[class_name]
            full_image_path = os.path.join(data_path, 'data', 'images', image_id + '.jpg')
            new_image_path = os.path.join(data_path, 'test', 'class_' + str(class_id).zfill(3), image_id + '.jpg')
            print('{} {}'.format(full_image_path, new_image_path))
            os.system('cp {} {}'.format(full_image_path, new_image_path))
