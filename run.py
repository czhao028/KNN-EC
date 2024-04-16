import os
import yaml
import time
import torch
import faiss 
from tqdm import tqdm
from pathlib import Path
import numpy as np
from utils.bertWhiteness import BertWhitening,transform_and_normalize,neg_softmax,softmax
from models.model import  BertForSequenceClassification
from sklearn.metrics import classification_report,f1_score,accuracy_score
from utils.dataprocessor import getTrainData,getTestData,saveTestResults
from transformers import BertTokenizer,get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from torch.utils.data import  DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import KFold
#from pykeops.torch import LazyTensor
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pickle

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train(config):
    np.random.seed(config['general']['seed'])
    torch.manual_seed(config['general']['seed'])
    torch.cuda.manual_seed_all(config['general']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpu_ids']
    
    model_name=config['model']['model_name'] # model_list=['bert-base-uncased','roberta-base','roberta-large']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset=getTrainData(tokenizer,model_name,config['data']['data_path'])
    #dev_data=getDevData(tokenizer,model_name,config['data']['data_path'])

    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['training']['train_batch_size'])
    # dev_sampler = SequentialSampler(dev_data)
    # dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=config['training']['dev_batch_size'])
    #

    num_folds = config['training']['num_folds']
    batch_size = config['training']['train_batch_size']
    num_epochs = config['training']['num_train_epochs']

    history_F1 = np.zeros(shape=(num_folds, num_epochs))
    history_trainingAcc = np.zeros(shape=(num_folds, num_epochs))
    history_validationAcc = np.zeros(shape=(num_folds, num_epochs))
    history_loss=np.zeros(shape=(num_folds, num_epochs))
    #train model
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    kfoldValidationPredictLabels = list()
    kfoldValidationTrueLabels = list()
    # Perform cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{num_folds}")

        # Split data into training and validation sets for this fold
        train_dataset = [dataset[i] for i in train_index]
        val_dataset = [dataset[i] for i in val_index]
        # Initialize model for each fold
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=config['model']['num_classes'],
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=eval(config['training']['learning_rate']),
                                      # args.learning_rate - default is 5e-5, our notebook had 2e-5
                                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                                      )
        total_steps = num_folds * len(train_dataloader) * config['training']['num_train_epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=config['training']['warmup_prop'],
                                                    # Default value in run_glue.py
                                                    num_training_steps=total_steps
                                                    )

        model.to(device)
        for epoch in range(num_epochs):
            # Train model for each epoch within this fold
            model.train()
            total_loss,step=0,0
            trainingAccuracyPerBatch = list()
            with tqdm(train_dataloader,
                              desc=f"Fold {fold + 1}/{num_folds} - Epoch {epoch + 1}/{num_epochs} - Training") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1}/{config['training']['num_train_epochs']}")
                    b_input_ids, b_input_mask,b_labels = batch[0].to(device),batch[1].to(device),batch[2].long().to(device)
                    model.zero_grad()

                    outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

                    loss = outputs[0]
                    total_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    predict=np.argmax(outputs[1].detach().cpu().numpy(), axis=1)
                    step+=1
                    training_acc = accuracy_score(batch[2].flatten(),predict.flatten())
                    trainingAccuracyPerBatch.append(training_acc)
                    tepoch.set_postfix(average_loss=total_loss/step,loss=loss.item(),f1=f1_score(batch[2].flatten(),predict.flatten(),average='weighted') ,accuracy='{:.3f}'.format(training_acc))
                    time.sleep(0.0001)
            history_trainingAcc[fold][epoch] = sum(trainingAccuracyPerBatch) / len(trainingAccuracyPerBatch)
            averageLossThisEpoch = total_loss / step
            history_loss[fold][epoch] = averageLossThisEpoch
            #eval model
            model.eval()
            #validation set accuracy/f1 score
            true_labels_validation,predict_labels_validation=[],[]
            for batch in val_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    outputs = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)
                logits = outputs[0].detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                predict_labels_validation.append(np.argmax(logits, axis=1).flatten())
                true_labels_validation.append(label_ids.flatten())
            true_labels_validation=[y for x in true_labels_validation for y in x]
            predict_labels_validation=[y for x in predict_labels_validation for y in x]
            kfoldValidationTrueLabels.extend(true_labels_validation)
            kfoldValidationPredictLabels.extend(predict_labels_validation)
            #print(classification_report(true_labels,predict_labels,digits=4))
            f1=f1_score(kfoldValidationTrueLabels,kfoldValidationPredictLabels,average='macro')
            history_F1[fold][epoch] = f1
            validationAccuracy = accuracy_score(kfoldValidationTrueLabels, kfoldValidationPredictLabels)
            history_validationAcc[fold][epoch] = validationAccuracy
            if config['training']['save_model'] and num_epochs > 0:
                torch.save(model, "{}/{}_epoch{}_f1{}.pt".format(config['data']['data_path'], model_name.replace("/", "-"),epoch, f1))
            print(f"F1 scores: {history_F1}")
            print(f"LOSSES: {history_loss}")
            print(f"TRAINING ACCURACIES: {history_trainingAcc}")
            print(f"VALIDATION ACCURACIES: {history_validationAcc}")
            plot_training_history(np.mean(history_loss,axis=0), train_acc=np.mean(history_trainingAcc,axis=0), val_acc=np.mean(history_validationAcc,axis=0), val_f1=np.mean(history_F1,axis=0))

def test(config):
    np.random.seed(config['general']['seed'])
    torch.manual_seed(config['general']['seed'])
    torch.cuda.manual_seed_all(config['general']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpu_ids']
    
    model_name=config['model']['model_name'] # model_list=['bert-base-uncased','roberta-base','roberta-large']
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model=torch.load(config['testing']['model_path'])
    model.to(device)
    
    test_data=getTestData(tokenizer,model_name,config['data']['data_path'])
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config['training']['test_batch_size'])
    
    #eval model        
    model.eval() 
    true_labels,predict_labels=[],[]
    for batch in test_dataloader:
        
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        logits = outputs[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predict_labels.append(np.argmax(logits, axis=1).flatten())
        true_labels.append(label_ids.flatten())
    
        
    true_labels=[y for x in true_labels for y in x]
    predict_labels=[y for x in predict_labels for y in x]
    print(classification_report(true_labels,predict_labels,digits=4))


def plot_training_history(train_losses, train_acc=None, val_acc=None, val_f1=None, filename='training_history.png'):
    print("Inside plot_training_history!")
    print(f"F1 scores: {val_f1}")
    print(f"LOSSES: {train_losses}")
    print(f"TRAINING ACCURACIES: {train_acc}")
    print(f"VALIDATION ACCURACIES: {val_acc}")
    plt.figure(figsize=(15, 5))

    # Plotting training loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plotting training accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    # Plotting F1 scores
    plt.subplot(1, 3, 3)
    if val_f1 is not None:
        plt.plot(val_f1, label='Validation F1 Score')
    if val_acc is not None:
        plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation Metrics')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(filename)

def findOptimalK(config):
    np.random.seed(config['general']['seed'])
    torch.manual_seed(config['general']['seed'])
    torch.cuda.manual_seed_all(config['general']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpu_ids']

    model_name = config['model']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = torch.load(config['testing']['model_path'])
    model.to(device)

    num_folds = config['training']['num_folds']
    batch_size = config['knnTest']['train_batch_size']
    # test_data=getTestData(tokenizer,model_name,config['data']['data_path'])
    train_data = getTrainData(tokenizer, model_name, config['data']['data_path'])
    test_data = getTrainData(tokenizer, model_name, config['data']['data_path'])

    # list_of_arrays = list()
    # i = 0
    # for batch in train_dataloader:
    #   train_input_ids, train_input_mask,train_labels = batch[0].to(device), batch[1].to(device),batch[2]#.long().to(device)
    #   #print(train_input_ids.shape)
    #   print(i)
    #   i+=1
    #   train_outputs = model(train_input_ids,
    #           token_type_ids=None,
    #           attention_mask=train_input_mask)
    #   train_batch_embeds = train_outputs[1].detach().cpu().numpy()
    #   list_of_arrays.append(train_batch_embeds)
    # train_embeds = np.concatenate(list_of_arrays, axis=0).astype("float32")

    # k = config['knnTest']['k']   # we want to see k nearest neighbors
    # alpha=config['knnTest']['alpha']
    k_values = np.arange(3, 20)
    heat_map = np.zeros(shape=(1, len(k_values)))
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    allf1scores = list()
    for j, k in enumerate(k_values):
        true_labels, predict_labels = [], []
        knn = KNeighborsClassifier(n_neighbors=k_values)
        f1scores = list()
        for fold, (train_index, val_index) in enumerate(kf.split(train_data)):

            train_dataset = [train_data[i] for i in train_index]
            val_dataset = [train_data[i] for i in val_index]
            train_sampler = SequentialSampler(train_dataset)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

            model.eval()
            train_embedds = list()
            num = 0
            # training embeddings
            for batch in train_dataloader:
                num += 1
                print(num)
                b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[
                    2]  # .long().to(device)
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

                train_embedd = outputs[1].detach().cpu().numpy()
                train_embedds.append(train_embedd)
                # predict_labels.append(np.argmax(res, axis=1).flatten())
                true_labels.append(b_labels.flatten())

            true_labels = np.array([y for x in true_labels for y in x])
            train_embeds = np.concatenate(train_embedds, axis=0).astype("float32")
            knn.fit(train_embeds, true_labels)
            # validation embeddings
            test_sampler = SequentialSampler(val_dataset)
            test_dataloader = DataLoader(val_dataset, sampler=test_sampler, batch_size=batch_size)
            test_embeddings_list = list()
            true_labels_test = list()
            for batch in test_dataloader:
                b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[
                    2]  # .long().to(device)
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)
                test_embedd = outputs[1].detach().cpu().numpy()
                test_embeddings_list.append(test_embedd)
                true_labels_test.append(b_labels.flatten())
            true_labels_test = np.array([y for x in true_labels_test for y in x])
            test_embeds = np.concatenate(test_embeddings_list, axis=0).astype("float32")
            predict_labels = knn.predict(test_embeds)

            print(classification_report(true_labels_test, predict_labels, digits=4))
            f1 = f1_score(true_labels_test, predict_labels, average="macro")

            f1scores.append(f1)
        avg_f1 = np.mean(np.array(f1scores))
        print(f"F1 score for k={k} is {avg_f1}")
        allf1scores.append(avg_f1)
    plt.plot(k_values, allf1scores)

    # Add title and labels
    plt.title('Effect of k in KNN on prediction accuracy')
    plt.xlabel('k-values')
    plt.ylabel('cross-validation accuracy')

    # Display the plot
    plt.show()

    # Save the plot
    plt.savefig('k_line_graph.png')
    # heat_map[1][j] = f1

    # plt.imshow(heat_map)
    # plt.title("F1 score on English training data")
    # plt.xlabel("k-value for KNN")
    # plt.ylabel("alpha hyperparameter")
    # plt.show()

def knnFit(config):
    np.random.seed(config['general']['seed'])
    torch.manual_seed(config['general']['seed'])
    torch.cuda.manual_seed_all(config['general']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpu_ids']

    model_name = config['model']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = torch.load(config['testing']['model_path'])
    model.to(device)

    batch_size = config['training']['train_batch_size']
    train_data = getTrainData(tokenizer, model_name, config['data']['data_path'])

    knn = KNeighborsClassifier(n_neighbors=config['knnTest']['k'])
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    model.eval()
    all_train_embedds = list()
    true_labels = []
    # training embeddings
    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[
            2]  # .long().to(device)
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

        train_embedd = outputs[1].detach().cpu().numpy()
        all_train_embedds.append(train_embedd)
        # predict_labels.append(np.argmax(res, axis=1).flatten())
        true_labels.append(b_labels.flatten())

    true_labels = np.array([y for x in true_labels for y in x])
    train_embeds = np.concatenate(all_train_embedds, axis=0).astype("float32")
    knn.fit(train_embeds, true_labels)
    knnPickle = open(config['knnTest']['model_file'] + config['knnTest']['k'], 'wb')
    # source, destination
    pickle.dump(knn, knnPickle)
    # close the file
    knnPickle.close()

def knnPredict(config):
    np.random.seed(config['general']['seed'])
    torch.manual_seed(config['general']['seed'])
    torch.cuda.manual_seed_all(config['general']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpu_ids']

    model_name = config['model']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = torch.load(config['testing']['model_path'])
    model.to(device)

    batch_size = config['knnTest']['test_batch_size']
    test_data = getTestData(tokenizer, model_name, config['data']['data_path'])

    knn = pickle.load(open(config['knnTest']['model_file'] + config['knnTest']['k'], 'rb'))
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    model.eval()
    all_test_embedds = list()
    # training embeddings
    for batch in test_dataloader:
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[
            2]  # .long().to(device)
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

        test_embedd = outputs[1].detach().cpu().numpy()
        all_test_embedds.append(test_embedd)

    test_embeds = np.concatenate(all_test_embedds, axis=0).astype("float32")
    predict_labels = knn.predict(test_embeds)
    print(predict_labels)
    saveTestResults(config['data']['data_path'] + "/test.tsv", predict_labels, config['knnTest']['prediction_results'])

def main():
    project_root: Path = Path(__file__).parent
    with open(str(project_root / "config.yml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    #train(config)
    knnFit(config)
    knnPredict(config)

if __name__ == '__main__':
    main()
    