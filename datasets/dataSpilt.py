import numpy as np
import torch
import random
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

from options import opt


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# -------------------------------------------------------------------------------------------------------
# IMAGE DATASET CLASS
# -------------------------------------------------------------------------------------------------------
class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        #if self.inputs
        return self.inputs.shape[0]


def get_MNIST():
    dataset_train = datasets.MNIST(root=opt.label_dir, train=True, download=True,
                                       transform = get_default_data_transforms(opt.dataset, verbose=False)[0])
    dataset_test = datasets.MNIST(root=opt.label_dir, train=False, download=True,
                                      transform = get_default_data_transforms(opt.dataset, verbose=False)[1])
    return dataset_train, dataset_test
    # return dataset_train.train_data.numpy(), dataset_train.train_labels.numpy(), dataset_test.test_data.numpy(), dataset_test.test_labels.numpy()


def get_CIFAR10():
    '''Return CIFAR10 train/test data and labels as numpy arrays'''
    data_train = datasets.CIFAR10(root=opt.label_dir, train=True, download=True)
    data_test = datasets.CIFAR10(root=opt.label_dir, train=False, download=True)
    #
    # x_train, y_train = data_train.train_data.transpose((0, 3, 1, 2)), np.array(data_train.train_labels)
    # x_test, y_test = data_test.test_data.transpose((0, 3, 1, 2)), np.array(data_test.test_labels)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)


    return x_train, y_train, x_test, y_test


def get_default_data_transforms(name, train=True, verbose=True):
    transforms_train = {
        # 'MNIST': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'MNIST': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'FashionMNIST': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((32, 32)),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'CIFAR10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),

    }
    transforms_eval = {
        'MNIST': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'FashionMNIST': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((3，32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'CIFAR10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
    }

    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train[name].transforms:
            print(' -', transformation)
        print()

    return transforms_train[name], transforms_eval[name]


def relabel_K(dataset_train, unlabel_dict):
    count = 0
    for index, label in enumerate(dataset_train.labels):
        if count < len(unlabel_dict) and index == unlabel_dict[count]:
            dataset_train.labels[index] += opt.num_classes
            count += 1
    return dataset_train


def puSpilt_index(dataset, indexlist, samplesize):

    labels = dataset.labels.numpy()

    labeled_size = 0
    for i in indexlist:
        labeled_size += int(samplesize[i] * opt.positiveRate)
    unlabeled_size = len(labels) - labeled_size

    #l_shard = [i for i in range(int(singleClass * pos_rate))]
    labeled = np.array([], dtype='int64')
    unlabeled = np.array([], dtype='int64')
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    priorlist = []

    # divide to unlabeled
    bias = 0
    for i in range(opt.num_classes):
        if samplesize[i] != 0:
            if i in indexlist and samplesize[i]>=40:
                labeled = np.concatenate(
                    (labeled, idxs[bias : int(bias + opt.positiveRate * samplesize[i])]), axis=0)
                bias += int(opt.positiveRate * samplesize[i])
                unlabeled = np.concatenate(
                    (unlabeled, idxs[bias : int(bias + (1-opt.positiveRate) * samplesize[i])]), axis=0)
                bias += int((1-opt.positiveRate) * samplesize[i])
                priorlist.append(samplesize[i] * (1 - opt.positiveRate) / unlabeled_size)
            else:
                unlabeled = np.concatenate((unlabeled, idxs[bias : bias + samplesize[i]]), axis=0)
                bias += samplesize[i]
                priorlist.append(samplesize[i] / unlabeled_size)
        else:
            priorlist.append(0.0)

    return labeled, unlabeled, priorlist

# -------------------------------------------------------------------------------------------------------
# SPLIT DATA AMONG CLIENTS
# -------------------------------------------------------------------------------------------------------
def split_image_data(data, labels, n_clients=10, classes_per_client=10, shuffle=True, verbose=True):
    '''
    Splits (data, labels) evenly among 'n_clients s.t. every client holds 'classes_per_client
    different labels
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    '''
    n_data = len(data)
    n_labels = np.max(labels) + 1

    data_per_client = [n_data // n_clients] * n_clients
    data_per_client_per_class = [data_per_client[0] // classes_per_client] * n_clients

    if sum(data_per_client) > n_data:
        print("Impossible Split")
        exit()

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []
        budget = data_per_client[i]
        c = max(c, 0)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

    if verbose:
        print_split(clients_split)
    return clients_split


def iid_partition(dataset, clients):
    """
    I.I.D paritioning of data over clients
    Shuffle the data
    Split it between clients

    params:
      - dataset (torch.utils.Dataset): Dataset containing the MNIST Images
      - clients (int): Number of Clients to split the data between

    returns:
      - Dictionary of image indexes for each client
    """

    num_items_per_client = int(len(dataset) / clients)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]

    for i in range(clients):
        client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
        image_idxs = list(set(image_idxs) - client_dict[i])
    import pdb;
    pdb.set_trace()
    return client_dict


def get_data_loaders(verbose=True):
    # x_train, y_train, x_test, y_test = globals()['get_' + opt.dataset]()
    dataset_train, dataset_test = globals()['get_' + opt.dataset]()

    transforms_train, transforms_eval = get_default_data_transforms(opt.dataset, verbose=False)

    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=opt.pu_batchsize, shuffle=True)


    # split = split_image_data(x_train, y_train, n_clients=opt.num_clients,
    #                          classes_per_client=opt.classes_per_client,
    #                          verbose=verbose)
    split = iid_partition(dataset_train, opt.num_clients)

    train_dataset = []
    priorlist = []
    indexlist = []  #防止返回值出错

    count = 0
    randomIndex_num = [4,4,3,3,2,2,1,1,1,1]

    for i, (x, y) in enumerate(split):
        indexList = []
        dataset = CustomImageDataset(x, y, transforms_train)
        selectcount = [0 * 1 for i in range(opt.num_classes)]

        # 计算每一类的样本量
        samplesize = [0 * 1 for i in range(opt.num_classes)]
        for l in dataset.labels:
            samplesize[l] += 1
        # if opt.dataset == 'MNIST'
        if opt.P_Index_accordance:          # indexlist长度一致
            for j in range(opt.randomIndex_num):
                k = 0
                while True:
                    index = (count + j + k) % opt.num_classes
                    if (i == (opt.num_clients - 1) or samplesize[index] > 40) and selectcount[index] < opt.randomIndex_num \
                            and (sum(m==0 for m in selectcount)>(opt.num_classes-opt.classes_per_client) and index not in indexList):
                        indexList.append(index)
                        selectcount[index] += 1
                        break
                    elif k > opt.num_classes:
                        break
                    k += 1
        else:
            for j in range(randomIndex_num[i]):
                k = 0
                while True:
                    index = (count + j + k) % opt.num_classes
                    if samplesize[index] > 40 and selectcount[index] < sum(randomIndex_num)/opt.num_classes and index not in indexList:
                        indexList.append(index)
                        selectcount[index] += 1
                        break
                    elif k > opt.num_classes:
                        break
                    k += 1

        label_dict, unlabel_dict, priorList = puSpilt_index(dataset, indexList, samplesize)
        priorlist.append(priorList)
        indexlist.append(indexList)
        unlabel_dict = np.sort(unlabel_dict)  # dict序列排序
        dataset = relabel_K(dataset, unlabel_dict)  # 将挑出的unlabeled数据标签全部改为classnum-1
        train_dataset.append(dataset)
        count += len(indexList)


    print(indexlist)

    client_loaders = [torch.utils.data.DataLoader(
            data, batch_size=opt.pu_batchsize, num_workers=16, shuffle=True) for data in train_dataset]

    stats = [x.shape[0] for x, y in split]

    return client_loaders, stats, test_loader, torch.Tensor(indexlist).cuda(), torch.Tensor(priorlist).cuda()



def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))