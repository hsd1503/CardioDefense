import torch
import numpy as np
from torch.utils.data import Dataset
MAX_SENTENCE_LENGTH = 9000


class ECGDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list, target_list):
        """
        @param data_list: list of newsgroup tokens
        @param target_list: list of newsgroup targets

        """
        self.data_list = data_list
        self.target_list = target_list
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """

        token_idx = self.data_list[key][:MAX_SENTENCE_LENGTH]   # 样本数据量最多到max_sentence,此处已经限制了样本的长度
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]   # 这里提取出了每一个样本的数据长度

class ECGDataset2(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list, length_list, target_list):
        """
        @param data_list: list of newsgroup tokens
        @param target_list: list of newsgroup targets

        """
        self.data_list = data_list
        self.target_list = target_list
        self.length_list = length_list
        assert (len(self.data_list) == len(self.target_list))
        assert (len(self.data_list) == len(self.length_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """

        token_idx = self.data_list[key][:MAX_SENTENCE_LENGTH]   # 样本数据量最多到max_sentence
        length = self.length_list[key]
        label = self.target_list[key]
        return [token_idx, length, label]   # 这里提取出了每一个样本的数据长度



def ecg_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    # print("collate batch: ", batch[0][0])
    # batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    # 在这里，data[0]表示数据，data[1]表示长度，data[2]表示标签
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    # padding
    for datum in batch:
        remainder = MAX_SENTENCE_LENGTH - datum[1]
        padded_vec = np.pad(np.array(datum[0]), (int(remainder / 2), remainder - int(remainder / 2)), 'constant',
                         constant_values=0)
        data_list.append(padded_vec)
    return [torch.from_numpy(np.array(data_list)).unsqueeze(-2).type(torch.FloatTensor), torch.LongTensor(length_list),
            torch.LongTensor(label_list)]