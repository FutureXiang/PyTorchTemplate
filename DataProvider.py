import h5py
from torch.utils.data import Dataset, DataLoader
'''
DataProvider:

    class MyDataset(Dataset):
        customize   __init__(self, ...)
        overwrite   __getitem__(self, index)
        overwrite   __len__(self)

    class DataProvider():
        customize   __init__(self, batch_size, ...):
            instantiate MyDataset

        define      build(self):
            instantiate DataLoader
            define      DataIter

        define      next(self):
            get         next batch
            regenerate  DataLoader (when epoch ends)
'''


class MyDataset(Dataset):
    def __init__(self,
                 trainFile,
                 testFile,
                 transform=None,
                 target_transform=None):
        train = h5py.File(trainFile, 'r')
        test = h5py.File(testFile, 'r')
        # label = 0, 1, 2, 3, 4, 5
        self.train_x = train['train_set_x'][:]
        self.train_y = train['train_set_y'][:]
        self.test_x = test['test_set_x'][:]
        self.test_y = test['test_set_y'][:]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.train_x[index]
        label = self.train_y[index]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        assert len(self.train_x) == len(self.train_y)
        return len(self.train_x)


class DataProvider:
    def __init__(self, batch_size, trainFile, testFile):
        self.batch_size = batch_size
        self.dataset = MyDataset(trainFile, testFile)
        self.dataiter = None

        self.train_len = self.dataset.__len__()
        self.train_batch_num = self.train_len // self.batch_size

    def build(self):
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     drop_last=True)
        self.dataiter = iter(self.dataloader)

    def next(self):
        if self.dataiter is None:
            self.build()
        try:
            '''
            1个epoch结束后，将会触发 StopIteration 的 Exception
            每个epoch重新生成 dataloader, 重新开始取batch
            '''
            batch = next(self.dataiter)
            return batch
        except StopIteration:
            self.build()
            batch = next(self.dataiter)
            return batch
