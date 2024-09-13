import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np
from tqdm import tqdm
from slippi_dataset import SlippiDataset
from configuration_dataset import DatasetConfig
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score, classification_report

from nn_list import *
import wandb
import gc
import os
os.environ["WANDB_DISABLED"] = "false"
hidden_size = 256

D = 0
freq = 1000


def train_loop(dataloader, model, optimizer, epoch, epochs, scheduler=None):
    actor, critic = model
    opt_act, opt_crt = optimizer
    sch_act, sch_crt = scheduler
    
    data_size = len(dataloader)
    pbar = tqdm(desc= f"epoch {epoch}/{epochs}", total = data_size, leave=True)
    mse = F.mse_loss
    bce = F.binary_cross_entropy
    ce = F.cross_entropy
    
    actor_cum_loss = 0
    critic_cum_loss = 0
    
    target_net = DeepResLSTMCritic(hidden_size, input_size, 2)
    target_net.load_state_dict(critic.state_dict())
    target_net.eval()
    target_net = target_net.to(device)
    actor.train()
    critic.train()
    for batch, data in enumerate(dataloader):
        X, y, R = data 
        del data
        X, y, R = X.to(device), y.to(device), R.to(device)
        
        init_state = actor.initial_state(X.shape[0], device)

        action_output, state = actor(torch.permute(X[:,:SEQ_LEN,:], (1, 0, 2)), init_state)
        actor_loss = ce(action_output.permute(1, 0, 2)[:, -1], y[:,D+SEQ_LEN,0].type(torch.LongTensor).to(device))
        
        # critic 
        with torch.no_grad():
            V_, _ = target_net(torch.permute(X[:,1:SEQ_LEN+1,:].float().to(device), (1, 0, 2)), init_state)
            target = V_.permute(1, 0, 2).float() + R[:, SEQ_LEN, 0].float()
            target = target[:,SEQ_LEN-1]
        V, _ = critic(torch.permute(X[:,:SEQ_LEN,:].float(), (1, 0, 2)), init_state)
        value_pred = V[SEQ_LEN-1, :].float()

        critic_loss = mse(value_pred, target)
        
        # 역전파
        opt_act.zero_grad()
        opt_crt.zero_grad()
        
        actor_loss.backward()
        critic_loss.backward()
        
        opt_act.step()
        opt_crt.step()

        actor_cum_loss = actor_loss.item()
        critic_cum_loss = critic_loss.item()
        

        pbar.set_postfix({"Actor Loss": round(actor_cum_loss, 8), "Critic Loss": round(critic_cum_loss, 8)})
        pbar.update(1)
        if batch % freq == freq-1:
            wandb.log({"Training Actor Loss" : actor_cum_loss/freq, "Training Critic Loss": critic_cum_loss/freq})
            
            actor_cum_loss = 0
            critic_cum_loss = 0
            target_net.load_state_dict(critic.state_dict())
        
        
        sch_act.step()
        sch_crt.step()
    pbar.close()
    return actor, critic

from sklearn.metrics import accuracy_score
from collections import defaultdict

def test_loop(dataloader, model, epoch, device):
    actor, critic = model
    
    # Initialize counters for true positives, false positives, false negatives, and support
    tp_buttons = defaultdict(lambda: defaultdict(int))
    fp_buttons = defaultdict(lambda: defaultdict(int))
    fn_buttons = defaultdict(lambda: defaultdict(int))
    support_buttons = defaultdict(lambda: defaultdict(int))

    tp_main_stick = defaultdict(int)
    fp_main_stick = defaultdict(int)
    fn_main_stick = defaultdict(int)    
    support_main_stick = defaultdict(int)

    tp_c_stick = defaultdict(int)
    fp_c_stick = defaultdict(int)
    fn_c_stick = defaultdict(int)
    support_c_stick = defaultdict(int)

    tp_trigger = defaultdict(int)
    fp_trigger = defaultdict(int)
    fn_trigger = defaultdict(int)
    support_trigger = defaultdict(int)
    
    actor_cum_loss = 0      
    critic_cum_loss = 0
    
    data_size = len(dataloader)
    pbar = tqdm(desc= f"Test: epoch {epoch+1}/{epochs}", total = data_size, leave=True)
    actor.eval()
    critic.eval()
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            X, y, R = data # ['p1_state'], data['p1_action'], data['p1_reward']
            
            init_state = actor.initial_state(X.shape[0], device)
            button_preds, main_stick_pred, c_stick_pred, trigger_pred, _ = actor(
                torch.permute(torch.concat([X[:,:SEQ_LEN,:], y[:,D:D+SEQ_LEN,:]], axis=2), (1, 0, 2)).float(), init_state)
            
             # Button loss 계산
            btn_loss = 0
            for idx, button_pred in enumerate(button_preds):
                button_pred = button_pred.permute(1, 0, 2)[:, :, 0]  
                # 입력과 타겟의 차원을 맞추기 위해 permute 사용
                ce_btn_loss = F.binary_cross_entropy(button_pred, 
                                                y[:, D+1:D+1+SEQ_LEN, idx])
                p_t = button_pred * y[:, D+1:D+1+SEQ_LEN, idx] + (1 - button_pred) * (1 - y[:, D+1:D+1+SEQ_LEN, idx])
                btn_loss += (ce_btn_loss * ((1 - p_t) ** gamma)).mean()

            # Main stick loss 계산
            main_stick_loss = F.cross_entropy(main_stick_pred.permute(1, 0, 2).reshape(-1, main_stick_pred.size(2)), 
                                            y[:, D+1:D+1+SEQ_LEN, 5].reshape(-1).type(torch.int64))

            # C-stick loss 계산
            c_stick_loss = F.cross_entropy(c_stick_pred.permute(1, 0, 2).reshape(-1, c_stick_pred.size(2)), 
                                        y[:, D+1:D+1+SEQ_LEN, 6].reshape(-1).type(torch.int64))

            # Trigger loss 계산
            trigger_loss = F.cross_entropy(trigger_pred.permute(1, 0, 2).reshape(-1, trigger_pred.size(2)), 
                                        y[:, D+1:D+1+SEQ_LEN, 7].reshape(-1).type(torch.int64))

            # 전체 loss 계산 및 평균화
            actor_loss = main_stick_loss + c_stick_loss + trigger_loss + btn_loss / idx   
            
            # critic 
            V_, _ = critic(torch.permute(torch.concat([X[:,1:SEQ_LEN+1,:].to(device), y[:,D+1:D+SEQ_LEN+1,:].to(device)], axis=2).float(), (1, 0, 2)), init_state)
            V, _ = critic(torch.permute(torch.concat([X[:,:SEQ_LEN,:].to(device), y[:,D:D+SEQ_LEN,:].to(device)], axis=2).float(), (1, 0, 2)), init_state)   
            value_pred = V.permute(1, 0, 2).reshape(-1, 1).float()
            target = V_.permute(1, 0, 2).reshape(-1, 1).float() + R[:, D+1:D+1+SEQ_LEN].reshape(-1, 1).float().to(device)
                
            critic_loss = F.mse_loss(value_pred, target)

            actor_cum_loss += actor_loss.item()
            critic_cum_loss += critic_loss.item()

            # Update true positive, false positive, and false negative counts
            for t in range(SEQ_LEN):
                for idx, button_pred in enumerate(button_preds):
                    true_labels = y[:,D+1+t, idx].type(torch.int64).cpu().numpy()
                    pred_labels = torch.argmax(button_pred[t, :, :], dim=1).cpu().numpy().flatten()

                    for true, pred in zip(true_labels, pred_labels):
                        support_buttons[idx][true] += 1  # Count the occurrence of each true label
                        if true == pred:
                            tp_buttons[idx][true] += 1
                        else:
                            fp_buttons[idx][pred] += 1
                            fn_buttons[idx][true] += 1

                true_main_stick = y[:,D+1+t, 5].cpu().numpy().flatten()
                pred_main_stick = torch.argmax(main_stick_pred[t, :, :], dim=1).cpu().numpy().flatten()
                for true, pred in zip(true_main_stick, pred_main_stick):
                    support_main_stick[true] += 1  # Count the occurrence of each true label
                    if true == pred:
                        tp_main_stick[true] += 1
                    else:
                        fp_main_stick[pred] += 1
                        fn_main_stick[true] += 1

                true_c_stick = y[:,D+1+t, 6].cpu().numpy().flatten()
                pred_c_stick = torch.argmax(c_stick_pred[t, :, :], dim=1).cpu().numpy().flatten()
                for true, pred in zip(true_c_stick, pred_c_stick):
                    support_c_stick[true] += 1  # Count the occurrence of each true label
                    if true == pred:
                        tp_c_stick[true] += 1
                    else:
                        fp_c_stick[pred] += 1
                        fn_c_stick[true] += 1

                true_trigger = y[:,D+1+t, 7].cpu().numpy().flatten()
                pred_trigger = torch.argmax(trigger_pred[t, :, :], dim=1).cpu().numpy().flatten()
                for true, pred in zip(true_trigger, pred_trigger):
                    support_trigger[true] += 1  # Count the occurrence of each true label
                    if true == pred:
                        tp_trigger[true] += 1
                    else:
                        fp_trigger[pred] += 1
                        fn_trigger[true] += 1
            freq = 50
            if batch % freq == freq-1:
                pbar.set_postfix({"Actor Loss": round(actor_cum_loss/batch, 8), "Critic Loss": round(critic_cum_loss/batch, 8)})
                pbar.update(min(freq, data_size - batch))
        
    wandb.log({"Test Actor Loss" : actor_cum_loss/len(dataloader), "Test Critic Loss": critic_cum_loss/len(dataloader)})

    pbar.close()
    
    # Calculate precision and recall for each category
    def calculate_metrics(tp, fp, fn, support):
        precision = {k: tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) > 0 else 0 for k in tp}
        recall = {k: tp[k] / (tp[k] + fn[k]) if (tp[k] + fn[k]) > 0 else 0 for k in tp}
        return precision, recall, support

    # Calculate precision, recall, and support for buttons
    button_metrics = {}
    for idx in range(6):  # Assuming there are 6 buttons
        precision, recall, support = calculate_metrics(tp_buttons[idx], fp_buttons[idx], fn_buttons[idx], support_buttons[idx])
        button_metrics[idx] = (precision, recall, support)

    main_stick_precision, main_stick_recall, main_stick_support = calculate_metrics(tp_main_stick, fp_main_stick, fn_main_stick, support_main_stick)
    c_stick_precision, c_stick_recall, c_stick_support = calculate_metrics(tp_c_stick, fp_c_stick, fn_c_stick, support_c_stick)
    trigger_precision, trigger_recall, trigger_support = calculate_metrics(tp_trigger, fp_trigger, fn_trigger, support_trigger)

    # Print metrics for each label
    # def print_metrics(label_name, metrics):
    #     print(f"\n{label_name} Metrics:")
    #     for label, (precision, recall, support) in metrics.items():
    #         print(precision.keys(), recall.keys(), support.keys())
    #         print(f"Label {label}: Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, Support: {support[0]}")

    # print_metrics("Button", button_metrics)
    # print_metrics("Main Stick", {0: (main_stick_precision, main_stick_recall, main_stick_support)})
    # print_metrics("C Stick", {0: (c_stick_precision, c_stick_recall, c_stick_support)})
    # print_metrics("Trigger", {0: (trigger_precision, trigger_recall, trigger_support)})
    
    return button_metrics   # 필요한 경우 다른 값들도 반환할 수 있습니다.


class DeviceException(Exception):
    def __init__(self):
        super().__init__('CUDA is not available')

def get_ckpts():
    ckpts = os.listdir("./models")
    ckpt_dict = dict()
    for ckpt in ckpts:
        if not ckpt.startswith("test1") or not ckpt.endswith(".pt"):
            continue
        epoch = int(ckpt.split("_")[1][5:])
        if epoch in ckpt_dict.keys():
            ckpt_dict[epoch].append(ckpt)
        else:
            ckpt_dict[epoch] = [ckpt]
    if len(ckpt_dict.keys()) == 0:
        return None, None
    resume_epoch = max(ckpt_dict.keys())
    resume_ckpts = ckpt_dict[resume_epoch]
    return resume_epoch, sorted(resume_ckpts)


import random

def load_data(config: DatasetConfig, train_ratio=1.):
    ## define train episode and validation episode
    if config.agent_name is not None:
        import json
        with open(os.path.join(config.basepath, "processed_replays.json"), "r") as f:
            data = json.load(f)
        total_datafile = []
        ## concat all maps
        for map in data[config.agent_name].keys():
            for i in data[config.agent_name][map]["0"]:
                total_datafile.append((i, 0))
            for i in data[config.agent_name][map]["1"]:
                total_datafile.append((i, 1))
            
    else:
        total_datafile = [os.path.join(config.basepath, name) for name in os.listdir(config.basepath) if name.endswith(".dat")]
    train_episodes = (random.sample(total_datafile, k=int(train_ratio*len(total_datafile))))
    val_episodes = (list(set(total_datafile) - set(train_episodes)))

    train_dataset = SlippiDataset(config=config, mm_filelist=train_episodes)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,  
        pin_memory=True, 
        drop_last=True, 
        num_workers=config.num_workers
    )
    if len(val_episodes) == 0:
        val_dataloader = None
    else:
        val_dataset = SlippiDataset(config=config, mm_filelist=val_episodes)
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            pin_memory=True,
            drop_last=True, 
            num_workers=config.num_workers
        )
    return train_dataloader, val_dataloader



if __name__ == "__main__":
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    if (USE_CUDA == False):
        print("CUDA NOT AVAILABLE, WOULD YOU CONTINUE?(y if yes, n if no)")
        command = ""
        while (command == 'y' or command == 'n'):
            command = input()
        if command == 'n':
            raise DeviceException
    else:
        print("current device:",device)
        
    
    
    train_ratio = 1.
    throw_away_ratio = 0.
    batch_size = 1024
    
    config = DatasetConfig()
    train_dataloader, test_dataloader = load_data(config)
    # throw_away_size = int(len(dataset) * throw_away_ratio)
    # train_size = int(train_ratio * (len(dataset)-throw_away_size))
    # test_size = len(dataset)-throw_away_size - train_size
    # if test_size != 0:
    #     train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, throw_away_size])
    # else:
    #     train_dataset = dataset
    #     test_dataset = None
    # print(f"train size: {train_size}, test size: {test_size}")

    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=3)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3) if test_dataset is not None else None
    # x, _, _ = train_dataset[10]
    # output_size = 8 # ABXYZ, main, c, trigger
    input_size = 960
    
    actor = DeepResLSTMActor(hidden_size, input_size, 2).to(device)
    critic = DeepResLSTMCritic(hidden_size, input_size, 2).to(device)
    
    print(f"Using {device} to train")
    epochs = 300
    
    gamma = 1
    learning_rate = 3e-4
    
    opt_act = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    sch_act = lr_scheduler.CosineAnnealingLR(opt_act, T_max=30000)
    
    opt_crt = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    sch_crt = lr_scheduler.CosineAnnealingLR(opt_crt, T_max=30000)
    
    import time
    model_name = "test1"# time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="melee_sl",
        id="swtlpo86",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "DeepResLSTM",
            "dataset": "ssbm-bot",
            "epochs": epochs,
            "gamma" : gamma,
            "model_name": model_name,
            "comment": "using target net, updated every 1000 batch"
        }
    )
    start_epoch, ckpts = get_ckpts() ## checkpoint
    if start_epoch is not None:
        actor.load_state_dict(torch.load(f"./models/{ckpts[0]}"))
        critic.load_state_dict(torch.load(f"./models/{ckpts[1]}"))
        for _ in range(1446*start_epoch):
            sch_act.step()
            sch_crt.step()
        print(ckpts)
    start_epoch = 0 if start_epoch is None else start_epoch+1
    for t in range(start_epoch, epochs):
        if t != 0:
            actor, critic = train_loop(train_dataloader, (actor, critic), (opt_act, opt_crt), epoch=t, epochs = epochs, scheduler=(sch_act, sch_crt))

        torch.save(actor.state_dict(), f"./models/{model_name}_epoch{t}_actor.pt")
        torch.save(critic.state_dict(), f"./models/{model_name}_epoch{t}_critic.pt")
            # acc_max = acc    
    print("Done!")    
