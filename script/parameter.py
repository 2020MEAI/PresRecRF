class Para:
    def __init__(self, lr, rec, drop, batch_size, epoch, dev_ratio, test_ratio,
                 embedding_dim, semantic, molecular, dataset):
        self.lr = lr
        self.rec = rec
        self.drop = drop
        self.batch_size = batch_size
        self.epoch = epoch
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.embedding_dim = embedding_dim
        self.semantic = semantic
        self.molecular = molecular
        self.dataset = dataset
