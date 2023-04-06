import numpy as np


def epoch(mode, device, net, dataloader, optimizer, criterion):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    if mode == "train":
        net.train()

    else:
        net.eval()

    for i, data in enumerate(dataloader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        n_b = labels.shape[0]

        outputs = net(imgs)
        loss = criterion(outputs, labels)

        acc = np.sum(np.equal(np.argmax(outputs.cpu().data.numpy(), axis=-1), labels.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg
