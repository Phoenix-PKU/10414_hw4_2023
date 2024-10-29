import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

device = ndl.cpu()

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim, device=device),
                norm(hidden_dim, device=device),
                nn.ReLU(),
                nn.Dropout(drop_prob, device=device),
                nn.Linear(hidden_dim, dim, device=device),
                norm(dim, device=device), 
            )
        ), 
        nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    linear_first = nn.Linear(dim, hidden_dim, device=device)
    relu = nn.ReLU()
    residual_blocks = []
    for i in range(num_blocks):
        residual_blocks.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    linear_last = nn.Linear(hidden_dim, num_classes, device=device)
    return nn.Sequential(
        linear_first,
        relu,
        *residual_blocks,
        linear_last,
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    if opt is not None:
        model.train()
        all_loss, all_error = [], 0.0
        for batch_x, batch_y in dataloader:
            out = model(batch_x.reshape((batch_x.shape[0], 784)))
            loss = loss_func(out, batch_y)
            all_loss.append(loss.numpy())
            all_error += np.sum(out.numpy().argmax(axis=1) != batch_y.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
        return all_error/len(dataloader.dataset), sum(all_loss)/len(all_loss)
    else:
        model.eval()
        all_error, all_loss = 0.0, []
        for i, batch in enumerate(dataloader):
            X, y = batch[0].reshape((batch[0].shape[0], -1)), batch[1]
            out = model(X)
            loss = loss_func(out, y)
            all_error += np.sum(out.numpy().argmax(axis=1) != y.numpy())
            all_loss.append(loss.numpy())

        return all_error/len(dataloader.dataset), sum(all_loss)/len(all_loss)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        data_dir + "/train-images-idx3-ubyte.gz", 
        data_dir + "/train-labels-idx1-ubyte.gz"
    )
    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, device=device)
    test_dataset = ndl.data.MNISTDataset(
        data_dir + "/t10k-images-idx3-ubyte.gz",
        data_dir + "/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = ndl.data.DataLoader(dataset=test_dataset)
    model = MLPResNet(784, hidden_dim=hidden_dim)
    new_optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_error, train_loss, test_error, test_loss = 0, 0, 0, 0
    for _ in range(epochs):
        start_time = time.time()
        train_error, train_loss = epoch(train_dataloader, 
            model, new_optimizer)
        print('epoch:', _, 'train_error:', train_error, 'train_loss:', \
                train_loss, 'time:', time.time() - start_time)

    test_error, test_loss = epoch(test_dataloader, model)
    return (train_error, train_loss, test_error, test_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_error, train_loss, test_error, test_loss = train_mnist(
        data_dir='../data/mnist', batch_size=16, hidden_dim=16)
    print("Train error:", train_error)
    print("Train loss:", train_loss)
    print("Test error:", test_error)
    print("Test loss:", test_loss)
