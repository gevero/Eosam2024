import torch


# the training loop
def train_direct(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y) + torch.mean(torch.pow(torch.diff(pred),2))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss


# the validation loop
def val_direct(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()

    val_loss /= num_batches
    print("\n")
    print(f"Validation loss: {val_loss:>8f} \n")

    return val_loss


# the training loop
def train_inverse(dataloader, model, loss_reg, loss_ce, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred_l, pred_m, pred_g = model(X)
        true_l = y[:,0].type(torch.LongTensor).to(device)
        true_m = y[:,1].type(torch.LongTensor).to(device)
        true_g = y[:,2:].to(device)
        # print(pred_l,pred_m,pred_g)
        # print(true_l,true_m,true_g)
        loss = loss_ce(pred_l, true_l) + loss_ce(pred_m, true_m) + loss_reg(pred_g, true_g) 

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss


# the validation loop
def val_inverse(dataloader, model, loss_reg, loss_ce, device):
    num_batches = len(dataloader)
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred_l, pred_m, pred_g = model(X)
            true_l = y[:,0].type(torch.LongTensor).to(device)
            true_m = y[:,1].type(torch.LongTensor).to(device)
            true_g = y[:,2:].to(device)
            loss = loss_ce(pred_l, true_l) + loss_ce(pred_m, true_m) + loss_reg(pred_g, true_g) 
            val_loss += loss.item()

    val_loss /= num_batches
    print("\n")
    print(f"Validation loss: {val_loss:>8f} \n")

    return val_loss