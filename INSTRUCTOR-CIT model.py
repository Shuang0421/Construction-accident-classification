
import random
import json
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, f1_score
from InstructorEmbedding import INSTRUCTOR
from focalloss import FocalLoss
import time
from utils import get_time_dif



class Config(object):
    # parameter configuration
    def __init__(self):
        self.pretrained_path_instructor = "pretrained-model/instructor-base"
        self.test_data_path = 'data/Test.csv'
        self.learning_rate = 1e-4
        self.weight_decay = 1e-2
        self.random_seed = 42
        self.dropout_rate = 0.5
        self.bs = 4
        self.require_improvement = 600
        self.num_epochs = 30
        self.train_itr = 20

        # Dataset 2 with 4770 samples

        self.data_path = "data/Accidents_Merge_clean_text_label.csv"
        self.output_dim = 7
        self.test_size = 0.4
        self.gamma = 4.0
        self.alpha = [0.5, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0]
        self.label2id = {
            '1 VIOLENCE AND OTHER INJURIES BY PERSONS OR ANIMALS': 0,
            '2 TRANSPORTATION INCIDENTS': 1,
            '3 FIRES AND EXPLOSIONS': 2,
            '4 FALLS,SLIPS,TRIPS': 3,
            '5 EXPOSURE TO HARMFUL SUBSTANCES OR ENVIRONMENTS': 4,
            '6 CONTACT WITH OBJECTS AND EQUIPMENT': 5,
            '7 OVEREXERTION AND BODILY REACTION': 6}

        # Dataset 1 with 1000samples

        # self.data_path = "data/tagged_Merge_clean_text_label.csv"
        # self.gamma = 3.0
        # self.alpha = [0.5, 0.75, 0.25, 1.0, 1.0, 0.5, 0.25, 0.25, 0.25, 1.0, 0.75]
        # self.output_dim = 11
        # self.test_size = 0.25
        # self.label2id = {
        #     'caught in/between objects': 0,
        #     'collapse of object': 1,
        #     'electrocution': 2,
        #     'exposure to chemical substances': 3,
        #     'exposure to extreme temperatures': 4,
        #     'falls': 5,
        #     'fires and explosion': 6,
        #     'struck by falling object': 7,
        #     'struck by moving objects': 8,
        #     'traffic': 9,
        #     'others': 10}

    def to_dict(self):
        # Outputs
        return {
            "pretrained_path_instructor": self.pretrained_path_instructor,
            "data_path": self.data_path,
            "output dim": self.output_dim,
            "learning rate": self.learning_rate,
            "weight decay": self.weight_decay,
            "random_seed": self.random_seed,
            "test size": self.test_size,
            "dropout rate": self.dropout_rate,
            "FocalLoss gamma": self.gamma,
            "FocalLoss alpha": self.alpha,
            "batch size": self.bs,
            "require improvement": self.require_improvement,
            "num epochs": self.num_epochs,
            "train itr": self.train_itr
        }

config = Config()
# Control random numbers to ensure output results
random.seed(config.random_seed)
np.random.seed(config.random_seed)
torch.manual_seed(config.random_seed)
torch.cuda.manual_seed_all(config.random_seed)

model = INSTRUCTOR(config.pretrained_path_instructor)


# Model define
class ClassificationModel(torch.nn.Module):
    def __init__(self, config):
        super(ClassificationModel, self).__init__()
        self.model = model
        self.config = config
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.fc = torch.nn.Linear(768, self.config.output_dim)

    def forward(self, input_ids, attention_mask):
        features = {'input_ids': input_ids, 'attention_mask': attention_mask}
        last_hidden_state = self.model(features)
        cls_token = last_hidden_state["sentence_embedding"]
        cls_token = self.dropout(cls_token)
        output = self.fc(cls_token)
        return output


# Data Load
class TextDataset(Dataset):
    def __init__(self, texts, labels, model):
        self.texts = texts
        self.labels = labels
        self.model = model

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.model.tokenizer.encode_plus(text, return_tensors='pt', max_length=512, padding='max_length',
                                                    truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label),
            'text_id': torch.tensor(idx),
            'text': self.texts[idx]
        }

    def __len__(self):
        return len(self.texts)


df = pd.read_csv(config.data_path)
label2id = config.label2id
df['label'] = df['label'].replace(label2id)
texts = df['text'].tolist()
labels = df['label'].tolist()


train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=config.test_size, stratify=labels,
                                                                    random_state=config.random_seed)

train_data_pd = pd.DataFrame(data={'text': train_texts, 'label': train_labels})
val_data_pd = pd.DataFrame(data={'text': val_texts, 'label': val_labels})
train_data_pd.to_csv("data/Train_data_pd.csv", index=True)
val_data_pd.to_csv('data/Val_data_pd.csv', index=True)


classify_model = ClassificationModel(config)
optimizer = torch.optim.AdamW(classify_model.parameters(), lr=config.learning_rate)
softmax = nn.Softmax(dim=1)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1000)


train_labels_tf = torch.tensor(train_labels)
class_sample_count = torch.tensor(
    [(train_labels_tf == t).sum() for t in torch.unique(train_labels_tf, sorted=True)])
weight = 1. / class_sample_count.float()
samples_weight = weight[train_labels_tf]

sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)


train_dataset = TextDataset(train_texts, train_labels, model)
val_dataset = TextDataset(val_texts, val_labels, model)

train_loader = DataLoader(train_dataset, batch_size=config.bs, shuffle=False, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=config.bs)


df_test = pd.read_csv(config.test_data_path)
test_texts = df_test['text'].tolist()
test_labels = df_test['label'].tolist()
test_dataset = TextDataset(test_texts, test_labels, model)
test_loader = DataLoader(test_dataset, batch_size=config.bs)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classify_model.to(device)


# evaluation
def evaluate(classify_model, val_loader, test=False):
    classify_model.eval()
    total_val_loss = 0
    val_preds = []
    val_preds_probas = []
    true_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = classify_model(input_ids, attention_mask)
            outputs_probas = softmax(outputs)

            # loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss = FocalLoss(gamma=config.gamma, alpha=config.alpha)(outputs, labels)
            total_val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            val_preds_probas.extend(outputs_probas.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    val_preds_probas = np.array(val_preds_probas)
    val_acc = accuracy_score(true_labels, val_preds)
    val_f1 = f1_score(true_labels, val_preds, average='weighted')

    if test:
        report = classification_report(true_labels, val_preds, digits=6)
        confusion = confusion_matrix(true_labels, val_preds)
        return val_acc, val_f1, total_val_loss / len(val_texts), report, confusion, val_preds, true_labels, val_preds_probas
    return val_acc, val_f1, total_val_loss / len(val_texts)


# test
def test(test_iter):
    start_time = time.time()
    classify_model = torch.load('result-model/save.pt')
    # print(classify_model)
    classify_model.eval()
    test_acc, test_f1, test_loss, test_report, test_confusion, preds, true, probas = evaluate(classify_model, test_iter,
                                                                                      test=True)

    print('time usage: ', time.time()-start_time)

    test_confusion_arr = np.array(test_confusion)
    config_dict = config.to_dict()
    print(config_dict)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test F1: {2:>6.2%}'
    print(msg.format(test_loss, test_acc, test_f1))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    preds_str = ' '.join([str(x) for x in preds])
    print('Preds:', preds)
    true_str = ' '.join([str(x) for x in true])
    print('True:', true)
    print('Probas:', probas)
    np.savetxt('result-model/Probas.txt', probas)
    df = pd.read_csv('data/Val_data_pd.csv')
    df['True'] = true
    df['Preds'] = preds
    df.to_csv('result-model/Val_data_pd_Preds.csv', index=False, encoding='utf-8')
    time_dif = get_time_dif(start_time)
    print("Test time usage:", time_dif)


    with open('result-model/Train_Record.txt', 'a+', encoding='utf-8') as f:
        content = (
                "\n" + '*' * 100 + "\n" +
                'Test time usage: ' + str(time.time()-start_time) + '\n' +
                'Test time usage: ' + str(time_dif) + '\n' +
                'Test Result...' + 'Hyperparameter: ' +
                json.dumps(config_dict) +
                "\n" + msg.format(test_loss, test_acc, test_f1) + "\n" +
                "Precision, Recall and F1-Score..." + "\n" + test_report + "\n" +
                "Confusion Matrix..." + "\n" + np.array2string(test_confusion_arr) + "\n" +
                "Preds:" + "\n" + preds_str + "\n" +
                "True:" + "\n" + true_str + "\n" +
                '*' * 100 + "\n"
        )
        f.write(content)


# Train
start_time = time.time()
total_batch = 0
total_epoch = 0
dev_best_loss = float('inf')
dev_best = float(0)
last_improve = 0
flag = False
dev_loss_figure = []

for epoch in range(config.num_epochs):
    classify_model.train()
    total_loss_train = 0
    train_preds = []
    train_true_labels = []

    print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = classify_model(input_ids, attention_mask)
        loss = FocalLoss(gamma=config.gamma, alpha=config.alpha)(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss_train += loss.item()
        true_labels_train = labels.data.cpu()
        preds_train = outputs.argmax(dim=1).cpu()
        train_preds.extend(preds_train.numpy())
        train_true_labels.extend(true_labels_train.numpy())
        train_acc = accuracy_score(true_labels_train, preds_train)
        train_f1 = f1_score(true_labels_train, preds_train, average='weighted')

        if total_batch % config.train_itr == 0:
            dev_acc, dev_f1, dev_loss= evaluate(classify_model, val_loader)
            if dev_acc > dev_best:
                dev_best = dev_acc
                torch.save(classify_model, 'result-model/save.pt')
                improve = '*'
                last_improve = total_batch
            else:
                improve = ''
            time_dif = time.time() - start_time

            msg = 'Iter: {0:>6},  Train Loss: {1:>5.4},  Train Acc: {2:>6.2%},  Train F1: {3:>6.2%}, ' \
                  'Val Loss: {4:>5.4},  Val Acc: {5:>6.2%},  Val F1: {6:>6.2%}, ' \
                  'Time: {7},  Improve: {8}'
            print(msg.format(total_batch, loss.item(), train_acc, train_f1,
                             dev_loss, dev_acc, dev_f1,
                             time_dif, improve))
            with open('result-model/Train_Record.txt', 'a+', encoding='utf-8') as f:
                f.write(msg.format(total_batch, loss.item(), train_acc, train_f1,
                                   dev_loss, dev_acc, dev_f1, time_dif, improve))
                f.write('\n')

        total_batch += 1

        if total_batch - last_improve > config.require_improvement:
            print("No optimization for a long time, auto-stopping...")
            flag = True
            break
        scheduler.step(dev_acc)

    if flag:
        break
    total_epoch += 1
    dev_acc_epoch, dev_f1_epoch, dev_loss_epoch = evaluate(classify_model, val_loader)

    dev_loss_figure.append(dev_loss_epoch)
    if dev_acc_epoch > dev_best:
        dev_best = dev_acc_epoch
        torch.save(classify_model, 'result-model/save.pt')
        improve = '*'
        last_improve = total_batch
    else:
        improve = ''
    time_dif = time.time() - start_time

    train_acc_epoch = accuracy_score(train_true_labels, train_preds)
    train_f1_epoch = f1_score(train_true_labels, train_preds, average='weighted')
    msg_epoch = 'Epoch: {0:>6},  Train Loss: {1:>5.4},  Train Acc: {2:>6.2%},  Train F1: {3:>6.2%}, ' \
                'Val Loss: {4:>5.4},  Val Acc: {5:>6.2%},  Val F1: {6:>6.2%}, ' \
                'Time: {7}, Improve: {8}'
    print(msg_epoch.format(epoch + 1, total_loss_train / len(train_loader), train_acc_epoch, train_f1_epoch,
                           dev_loss_epoch, dev_acc_epoch, dev_f1_epoch,
                           time_dif, improve))
    with open('result-model/Train_Record.txt', 'a+', encoding='utf-8') as f:
        f.write(msg_epoch.format(epoch + 1, total_loss_train / len(train_loader), train_acc_epoch, train_f1_epoch,
                                 dev_loss_epoch, dev_acc_epoch, dev_f1_epoch, time_dif, improve))
        f.write('\n')

# Model test
test(val_loader)