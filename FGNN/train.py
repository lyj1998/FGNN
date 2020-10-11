import torch
import pickle
import os
import datetime

from dataloader import MultiSessionsGraph
from torch_geometric.data import DataLoader
from model import FGNN
from metric import cal_hr, cal_mrr, cal_ndcg


def generate_config(dropouts, lrs, regs, config):
    configs = []
    for dropout in dropouts:
        for lr in lrs:
            for reg in regs:
                new_config = config.copy()
                new_config["dropout"] = dropout
                new_config["lr"] = lr
                new_config["reg"] = reg
                configs += [new_config]
    return configs


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset = "retailrocket_3"
train_dataset = MultiSessionsGraph(f"../../baseline/data/{dataset}", phrase='train')
test_dataset = MultiSessionsGraph(f"../../baseline/data/{dataset}", phrase='test')

config = {
    "item_num": 36969,  # item的最大序号  dn:40841 rr:36969
    "hidden_size": 100,
    "WGAT_heads": 8,
    "dropout": 0,
    "lr": 2e-3,
    "lr_decay": 0.1,
    "lr_decay_epoch": 4,
    "leaky_relu": 0.2,
    "reg": 1e-5,
    "batch_size": 100,
    "epoch": 12,
    "use_cuda": True,
    "metrics": [5, 10, 20]
}
if config["use_cuda"]:
    device = torch.device("cuda")
    config["device"] = device
else:
    device = torch.device("cpu")
    config["device"] = device

configs = generate_config(dropouts=[0], lrs=[2e-3, 3e-3], regs=[1e-4, 2e-4, 1e-5, 2e-5, 5e-5],
                          config=config)

training_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
testing_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

for config in configs:
    model = FGNN(config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["reg"], amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_decay_epoch"], gamma=config["lr_decay"])
    criterion = torch.nn.CrossEntropyLoss()
    print(config)

    best_hr = 0
    best_mrr = 0
    best_ndcg = 0
    total_loss = []
    for epoch in range(config["epoch"]):
        print("epoch:(%d/%d)" % (epoch + 1, config["epoch"]))
        print("Start Training:" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        scheduler.step()
        # train
        model.train()
        cur_loss = 0
        for i, data in enumerate(training_dataloader):
            optimizer.zero_grad()

            # forward & backward
            outputs = model(data.to(device))
            labels = data.y - 1
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            cur_loss += loss.item() * len(data)
            if i % 2000 == 0 and i != 0:
                print('(%d/%d) loss: %f' % (i, len(training_dataloader), loss.item()))

        print(f"total loss: {cur_loss / len(training_dataloader.dataset)}")
        total_loss += [cur_loss]

        # test
        model.eval()
        test_loss = 0
        print("Start testing:" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        with torch.no_grad():
            hrs = [0 for _ in range(len(config["metrics"]))]
            mrrs = [0 for _ in range(len(config["metrics"]))]
            ndcgs = [0 for _ in range(len(config["metrics"]))]
            for data in testing_dataloader:
                # forward & backward
                outputs = model(data.to(device))
                labels = data.y - 1
                loss = criterion(outputs, labels.to(device))

                test_loss += loss.item() * len(data)

                # metric
                result = torch.topk(outputs, k=config["metrics"][-1], dim=1)[1]
                for i, k in enumerate(config["metrics"]):
                    hrs[i] += cal_hr(result[:, :k].cpu().numpy(), labels.cpu().numpy())
                    mrrs[i] += cal_mrr(result[:, :k].cpu().numpy(), labels.cpu().numpy())
                    ndcgs[i] += cal_ndcg(result[:, :k].cpu().numpy(), labels.cpu().numpy())

            test_loss = test_loss / len(testing_dataloader.dataset)

            for i, k in enumerate(config["metrics"]):
                hrs[i] = hrs[i] / len(testing_dataloader.dataset)
                mrrs[i] = mrrs[i] / len(testing_dataloader.dataset)
                ndcgs[i] = ndcgs[i] / len(testing_dataloader.dataset)
                print(f'HR@{k}: {hrs[i]:.4f} MRR@{k}: {mrrs[i]:.4f} NDCG@{k}: {ndcgs[i]:.4f}')

            if hrs[-1] > best_hr:
                best_hr = hrs[-1]
                best_mrr = mrrs[-1]
                best_ndcg = ndcgs[-1]
                # torch.save(model, f"./models/CoSAN_{dataset}_"
                #                   f"dropout_{config['dropout']}_"
                #                   f"lr_{config['lr']}_"
                #                   f"reg_{config['reg']}_"
                #                   f"embedding_dim_{config['embedding_dim']}_"
                #                   f"k_{config['k']}_"
                #                   f"session_length_{config['session_length']}.pt")

                for i, k in enumerate(config["metrics"]):
                    print(f'best ever HR@{k}: {hrs[i]:.4f} MRR@{k}: {mrrs[i]:.4f} NDCG@{k}: {ndcgs[i]:.4f}')
            print('================================')
