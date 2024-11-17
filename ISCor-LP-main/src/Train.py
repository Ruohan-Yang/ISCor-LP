import os
import shutil
import torch
from tqdm import trange

def train_phase(model, train_loader, valid_loader, best_valid_path, args, log):
    model_path = 'save/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    best_valid_metric = 0
    best_model_epoch = 0
    patience = args.patience
    current_patience = 0

    for epoch in trange(args.epochs, desc='Training'):
        if args.ifDecay:
            p = epoch / (args.epochs - 1)
            learning_rate = args.learning_rate / pow((1 + 10 * p), 0.75)
        else:
            learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

        model.train()
        for data in train_loader:
            data = data[0]
            whole_loss = model.loss(data)
            whole_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        valid_acc, valid_pre, valid_f1, valid_auc = model.metrics_eval(valid_loader)
        # write_infor = "Valid acc:{:.4f}, Valid pre:{:.4f}, Valid f1:{:.4f}, Valid auc:{:.4f} ".format(valid_acc, valid_pre, valid_f1, valid_auc)
        # print(write_infor)

        # Update best_valid_auc and early stopping counter
        # Save the best model
        if valid_auc > best_valid_metric:
            best_valid_metric = valid_auc
            current_patience = 0
            torch.save(model.state_dict(), best_valid_path)
            best_model_epoch = epoch
        else:
            current_patience += 1

        # Check if early stopping conditions are met
        if args.EarlyStop and current_patience >= patience:
            print("Early stopping!")
            break

    write_infor = 'Best Epoch:[{}/{}]'.format(best_model_epoch+1, args.epochs)
    print(write_infor)
    log.write(write_infor + '\n')
    return best_valid_path

def test_phase(model, best_valid_dir, test_loader, log):
    # Load the best model for testing
    print('Load best model ' + best_valid_dir + ' ... ')
    model.load_state_dict(torch.load(best_valid_dir))
    model.eval()
    acc, pre, f1, auc = model.metrics_eval(test_loader)
    write_infor = "Test acc:{:.4f}, pre:{:.4f}, f1:{:.4f}, auc:{:.4f}".format(acc, pre, f1, auc)
    print(write_infor)
    log.write(write_infor + '\n')

def run_model(train_loader, valid_loader, test_loader, model, layer, args, log):
    best_valid_path = 'save/' + args.dataset + '_layer' + str(layer) + '_best_model.pth'
    if args.onlyTest and os.path.exists(best_valid_path):
        print('onlyTest')
        log.write('onlyTest\n')
    else:
        print('best_model.pth does not exist')
        print('train phase')
        best_valid_path = train_phase(model, train_loader, valid_loader, best_valid_path, args, log)
    test_phase(model, best_valid_path, test_loader, log)

