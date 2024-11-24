import sys
import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
import parameter
import time
from model import PresRecRF
from utils import *
base_dir = 'D:/Work/Project/Github/PresRecRF'


seed = 2022

np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

params = parameter.Para(
    lr=7e-4, rec=7e-4, drop=0.1, batch_size=32, epoch=30, dev_ratio=0.0, test_ratio=0.2, embedding_dim=256,
    semantic='Bert', molecular='HSP', dataset='Lung'
)
epsilon = 1e-13


out_name = f'PresRecRF_{params.dataset}_ba{params.batch_size}_ep{params.epoch}_lr{params.lr}_241022'


type = sys.getfilesystemencoding()
sys.stdout = Logger(f'{base_dir}/log/{out_name}.txt')

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
print("lr:", params.lr, "rec:", params.rec, "dropout:", params.drop, "batch_size:",
      params.batch_size, "epoch:", params.epoch, "dev_ratio:", params.dev_ratio, "test_ratio:", params.test_ratio)


nwfe = pd.read_excel(base_dir + '/data/Lung_case.xlsx')
data_amount = nwfe.shape[0]

nwfe_list = [x for x in range(data_amount)]
x_train, x_test = train_test_split(nwfe_list, test_size=(params.dev_ratio + params.test_ratio), shuffle=False,
                                   random_state=2022)
print("train_size:", len(x_train), "test_size:", len(x_test))

symptom_len = 1804
herb_len = 410

sym_list = [[0] * symptom_len for _ in range(data_amount)]
sym_array = np.array(sym_list)

herb_list = [[0] * herb_len for _ in range(data_amount)]
herb_array = np.array(herb_list)

herb_dosage_list = [[0.0] * herb_len for _ in range(data_amount)]
herb_dosage_array = np.array(herb_dosage_list)

for i in tqdm.tqdm(range(data_amount)):
    # sym
    sym_indexes = list(map(int, nwfe.iloc[i, 4].split(',')))
    sym_indexes = [x for x in sym_indexes if x < symptom_len]
    sym_array[i, sym_indexes] = 1

    # herb with dosage
    herbs = nwfe.iloc[i, 8].split(',')  # 先以药物为单位切分，切分后每个药物是“herbID|dosage”
    herb_IDs = [int(i.split('|')[0]) for i in herbs]
    herb_dosages = [float(i.split('|')[1]) for i in herbs]

    herb_array[i, herb_IDs] = 1
    herb_dosage_array[i, herb_IDs] = herb_dosages


train_dataset = PreDatasetDosage(sym_array[x_train], herb_array[x_train], herb_dosage_array[x_train])
test_dataset = PreDatasetDosage(sym_array[x_test], herb_array[x_test], herb_dosage_array[x_test])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size)

model = PresRecRF(params.batch_size, params.embedding_dim, symptom_len, herb_len, params.drop, params.semantic,
                  params.molecular)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
dosage_loss = torch.nn.MSELoss()  # mse

optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.rec)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.8)
early_stopping = EarlyStopping(patience=7, verbose=True)

train_loss_list = []
test_loss_list = []

out = pd.DataFrame(columns=['sym', 'herb', 'pre-herb10', 'pre-herb20', 'cos10', 'cos20'])

for epoch in range(params.epoch):
    model.train()
    training_loss = 0.0
    for i, (sid, hid, hd) in enumerate(train_loader):
        sid, hid, hd = sid.float(), hid.float(), hd.float()
        optimizer.zero_grad()
        pre_herb, pre_herb_dosage = model(sid)
        loss = criterion(pre_herb, hid) + dosage_loss(pre_herb_dosage, hd)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    print('[Epoch {}]train_loss: '.format(epoch + 1), training_loss / len(train_loader))
    train_loss_list.append(training_loss / len(train_loader))


torch.save(model.state_dict(), f'{base_dir}/checkpoint/checkpoint_{out_name}.pt')
# model.load_state_dict(torch.load(f'{base_dir}/checkpoint/checkpoint_{out_name}.pt'))
model.eval()

# 原测试指标
test_size = len(test_loader.dataset)
test_loss = 0
test_p5 = 0
test_p10 = 0
test_p20 = 0
test_r5 = 0
test_r10 = 0
test_r20 = 0
test_f1_5 = 0
test_f1_10 = 0
test_f1_20 = 0

# 回归相关函数
# mse
mse_criterion = nn.MSELoss()
# mae
mae_criterion = nn.L1Loss()
# huber
huber_criterion = nn.SmoothL1Loss()

mse_loss = 0
mae_loss = 0
huber_loss = 0

for i, (ttsid, tthid, tthdid) in enumerate(test_loader):
    ttsid, tthid, tthdid = ttsid.float(), tthid.float(), tthdid.float()
    ttpre_herb, ttpre_herb_dosage = model(ttsid)
    temp_ttloss = criterion(ttpre_herb, tthid) + dosage_loss(ttpre_herb_dosage, tthdid)
    test_loss += temp_ttloss.item()
    mse_loss += mse_criterion(ttpre_herb_dosage, tthdid).item()
    mae_loss += mae_criterion(ttpre_herb_dosage, tthdid).item()
    huber_loss += huber_criterion(ttpre_herb_dosage, tthdid).item()

    for i, hid in enumerate(tthid):
        trueLabel = []
        for idx, val in enumerate(hid):
            if val == 1:
                trueLabel.append(idx)
        ttsym = ttsid.numpy()[i]

        sym_check = []
        for sym in range(1804):
            if ttsym[sym] == 1:
                sym_check.append(sym)

        top5 = torch.topk(ttpre_herb[i], 5)[1]
        count = 0
        for m in top5:
            if m in trueLabel:
                count += 1
                # TP5 += 1

        test_p5 += float(count / 5)
        test_r5 += float(count / len(trueLabel))

        top10 = torch.topk(ttpre_herb[i], 10)[1]
        count = 0
        for m in top10:
            if m in trueLabel:
                count += 1
                # TP10 += 1
        test_p10 += count / 10
        test_r10 += count / len(trueLabel)

        top20 = torch.topk(ttpre_herb[i], 20)[1]
        count = 0
        for m in top20:
            if m in trueLabel:
                count += 1
        test_p20 += count / 20
        test_r20 += count / len(trueLabel)

test_p5 = test_p5 / test_size
test_p10 = test_p10 / test_size
test_p20 = test_p20 / test_size
test_r5 = test_r5 / test_size
test_r10 = test_r10 / test_size
test_r20 = test_r20 / test_size
test_f1_5 = 2 * test_p5 * test_r5 / (test_p5 + test_r5)
test_f1_10 = 2 * test_p10 * test_r10 / (test_p10 + test_r10)
test_f1_20 = 2 * test_p20 * test_r20 / (test_p20 + test_r20)

print(test_p5, test_p10, test_p20, test_r5, test_r10, test_r20, test_f1_5, test_f1_10, test_f1_20)

test_loss_list.append(test_loss / len(test_loader))
print('test_loss: '.format(epoch + 1), test_loss / len(test_loader))
print('Average loss of dosage: '
      'MSE: ', mse_loss / len(test_loader),
      'MAE: ', mae_loss / len(test_loader),
      'Huber: ', huber_loss / len(test_loader)
      )
