# -*- coding:utf-8 -*-
import random
import numpy as np

bands = 448  # number of spectral channels in raw data
spectral_start = 3  # 400nm channel index in wavelength list
spectral_end = 444  # 1000nm channel index in wavelength list
spectral_step = 13

class_num = 5
image_size = 100
# from 400nm to 1000nm, roughly spectral_step nm apart, all approximations
img_channels = int(np.ceil((spectral_end - spectral_start + 1) / spectral_step))

class_name = {0: "Almond",
              1: "Amber",
              2: "Shell",
              3: "Stone",
              4: "Woodchip"}

total_folders = 16
#prefix = "C:/Data/"
#prefix = "/srv/home/jiayu/539project/"
prefix = "./"
# path of folders
fname_in_folders = {0: prefix + "Almonds_II/Almond_1_2018-03-06_17-53-46/capture/Almond_1_2018-03-06_17-53-46_00",
                    1: prefix + "Almonds_II/Almond_2_2018-03-06_17-55-33/capture/Almond_2_2018-03-06_17-55-33_00",
                    2: prefix + "Almonds_II/Almond_3_2018-03-06_17-57-54/capture/Almond_3_2018-03-06_17-57-54_00",
                    3: prefix + "Almonds_II/Almond_4_2018-03-06_18-00-12/capture/Almond_4_2018-03-06_18-00-12_00",
                    4: prefix + "Almonds_II/Almond_5_2018-03-06_18-03-30/capture/Almond_5_2018-03-06_18-03-30_00",
                    5: prefix + "Almonds_II/Amber_1_2018-03-06_18-46-42/capture/Amber_1_2018-03-06_18-46-42_00",
                    6: prefix + "Almonds_II/Amber_2_2018-03-06_18-48-33/capture/Amber_2_2018-03-06_18-48-33_00",
                    7: prefix + "Almonds_II/Amber_3_2018-03-06_18-50-45/capture/Amber_3_2018-03-06_18-50-45_00",
                    8: prefix + "Almonds_II/Shells_1_2018-03-06_18-58-59/capture/Shells_1_2018-03-06_18-58-59_00",
                    9: prefix + "Almonds_II/Shells_2_2018-03-06_19-00-19/capture/Shells_2_2018-03-06_19-00-19_00",
                    10: prefix + "Almonds_II/Shells_3_2018-03-06_19-02-42/capture/Shells_3_2018-03-06_19-02-42_00",
                    11: prefix + "Almonds_II/Stones_1_2018-03-06_18-34-34/capture/Stones_1_2018-03-06_18-34-34_00",
                    12: prefix + "Almonds_II/Stones_2_2018-03-06_18-38-53/capture/Stones_2_2018-03-06_18-38-53_00",
                    13: prefix + "Almonds_II/Stones_3_2018-03-06_18-40-27/capture/Stones_3_2018-03-06_18-40-27_00",
                    14: prefix + "Almonds_II/WoodChips_1_2018-03-06_18-53-54/capture/WoodChips_1_2018-03-06_18-53-54_00",
                    15: prefix + "Almonds_II/WoodChips_2_2018-03-06_18-55-03/capture/WoodChips_2_2018-03-06_18-55-03_00"}

# number of files to process in each folder
num_files_in_folder = {0: 42,
                       1: 42,
                       2: 42,
                       3: 42,
                       4: 42,
                       5: 40,
                       6: 40,
                       7: 40,
                       8: 36,
                       9: 36,
                       10: 36,
                       11: 42,
                       12: 42,
                       13: 42,
                       14: 20,
                       15: 20}
total_files = 0
for v in num_files_in_folder.values():
    total_files += v

imgs_index_in_classes = {0: (0, 209),
                         1: (210, 329),
                         2: (330, 437),
                         3: (438, 563),
                         4: (564, 603)}

def load_data():
    global image_size, img_channels
    print("======Loading data======")

    # process all images in all folders
    # convert every one into a row vector (np array) with length 340,000

    data_all = np.zeros((total_files, img_channels, image_size, image_size))
    num = 0  # the index of curr img in data_all, 1st indexing param
    for i in range(total_folders):
        for j in range(num_files_in_folder[i]):
            print("Processing Image {ind}......".format(ind=num + 1))
            # Read raw data
            fname = fname_in_folders[i]
            if j < 9:
                fname += "0"
            fname += str(j + 1) + ".raw"
            img = np.fromfile(file=fname, dtype=np.uint16)

            # populate the data_all array
            # the data of every img is ordered by lines
            # each line is ordered by spectral channels
            # here, first 100 numbers is the data of first spectral channel
            # of the first entire line (100 pixel long) of the image, second
            # 100 is the data of second spectral channel ... first entire line...

            # for every line of of img
            for m in range(image_size):
                # data start index of curr line of img
                line_st = m * bands * image_size
                channel = 0  # spectral indexing
                # for every spectral channel that we want
                for n in range(spectral_start, spectral_end, spectral_step):
                    # data start and end index of curr spectral channel
                    st = n * image_size + line_st
                    end = st + image_size
                    data_all[num, channel, m, :] = img[st:end]
                    channel += 1
            num += 1
    # reshape data
    data_all = np.transpose(data_all, (0, 2, 3, 1))

    # create a np array for labels
    c = 0
    labels = np.zeros((1,class_num))
    for i in range(total_folders):
        if i == 5 or i == 8 or i == 11 or i == 14:
            c += 1
        labels = np.append(labels, np.array([[float(j == c) for j in range(class_num)] for _ in range(num_files_in_folder[i])]), axis=0)
    labels = labels[1:]

    print("Data :", np.shape(data_all), np.shape(labels))
    print("======Load finished======")
    # print(labels)

    return data_all, labels


def shuffle_data(data, labels):
    print("======Shuffling data======")
    # shuffle data
    #
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]

    print("======Shuffling finished======")

    return data, labels


def spectral_preprocessing(data):
    print("======Spectral preprocessing======")
    data = data.astype('float32')

    for i in range(img_channels):
        print("Channel {ind}......".format(ind=i + 1))
        data[:, :, :, i] = (data[:, :, :, i] - np.mean(data[:, :, :, i])) / np.std(data[:, :, :, i])

    print("======Spectral preprocessing finished======")

    return data


# ========================================================== #
# ├─ _random_crop()
# ├─ _random_flip_leftright()
# └─ data_augmentation()
# ========================================================== #

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [100, 100], 12)
    return batch
