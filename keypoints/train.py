from pathlib import Path
import torch
import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


import data
import model

def save_state(name, epoch, encoder, generator, optim, sched=None):
    data = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "generator": generator.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched
    }
    torch.save(data, name)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d0 = data.RawScan(
        path=Path('/home/kyle/Downloads/hsi/SPI_20240213125428_narrow_no_boson_cals_in.hdr'),
        wsz=512,
        stepsz=(32, 64)
    )
    d1 = data.RawScan(
        path=Path('/home/kyle/Downloads/hsi/SPI_20240213130043_olivine_narrow2.hdr'),
        wsz=512,
        stepsz = (32, 64),
    )
    d2 = data.RawScan(
        path=Path('/home/kyle/Downloads/hsi/SPI_20240620160920_0deg_skew.hdr'),
        wsz=512,
        stepsz = (32, 64),
    )
    d3 = data.RawScan(
        path=Path('/home/kyle/Downloads/hsi/SPI_20240620161551_30deg_skew.hdr'),
        wsz=512,
        stepsz = (32, 64),
    )
    d4 = data.RawScan(
        path=Path('/home/kyle/Downloads/hsi/SPI_20250221171722_[AIRBORNE]_Az113_MID.hdr'),
        wsz=512,
        stepsz = (32, 64),
    )

    d5 = data.RawScan(
        path=Path('/home/kyle/Downloads/hsi/SPI_20250228180448_[AIRBORNE]_LONGSCAN_MIDFOV_2X.hdr'),
        wsz=512,
        stepsz = (32, 64),
    )

    dataset = torch.utils.data.ConcatDataset([d0, d1, d2, d3, d4, d5,])

    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=True)

    restore = False
    points = 25
    lr = 2e-3
    epochs = 100

    encoder  = model.GeomEncoder(points).to(device)
    generator = model.Generator(points).to(device)
    optim = torch.optim.AdamW(list(encoder.parameters()) + list(generator.parameters()), lr=lr, weight_decay=5e-4)
    if restore:
        z = torch.load("checkpoint_25.pth", weights_only=False)
        encoder.load_state_dict(z["encoder"])
        generator.load_state_dict(z["generator"])
        optim.load_state_dict(z["optim"])

    encoder.train()
    generator.train()
    criterion = torch.nn.MSELoss()

    print(device)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    losses = []
    for epoch in range(epochs):
        for i, frames in enumerate(tqdm(dataloader)):
            frames = frames.to(device)
            optim.zero_grad()
            x = frames[:, [0], ...]
            x_prime = frames[:, [1], ...]

            hm, hg = encoder(x_prime)
            pred = generator(x, hg)

            if (epoch % 10 == 0) and (i == 0):
                fig, axes = plt.subplots(2, 5)
                axes[0][0].imshow(hm[0, 0].detach().cpu())
                axes[0][1].imshow(hm[0, 1].detach().cpu())
                axes[0][2].imshow(hm[0, 2].detach().cpu())
                axes[0][3].imshow(hm[0, 3].detach().cpu())
                axes[0][4].imshow(hm[0, 4].detach().cpu())

                axes[1][0].imshow(hm[0, 5].detach().cpu())
                axes[1][1].imshow(hm[0, 6].detach().cpu())
                axes[1][2].imshow(hm[0, 7].detach().cpu())
                axes[1][3].imshow(hm[0, 8].detach().cpu())
                axes[1][4].imshow(hm[0, 9].detach().cpu())

                fig1, axes1 = plt.subplots(2, 2)
                axes1[0][0].imshow(x[0].cpu().detach().squeeze())
                axes1[0][1].imshow(x_prime[0].cpu().detach().squeeze())
                axes1[1][0].imshow((x[0] - x_prime[0]).abs().cpu().detach().squeeze())
                axes1[1][1].imshow(pred[0].cpu().detach().squeeze())
                plt.show()

                model.viz_pts(x_prime[[0]].detach().cpu(), hm[[0]].detach().cpu())

            loss = criterion(pred, x_prime)
            losses.append(loss.item())
            loss.backward()
            optim.step()

        if epoch % 5 == 0:
            save_state(f"checkpoint_{epoch}.pth", epoch, encoder, generator, optim, None)
        # scheduler.step()

        print("Loss for epoch {}: {}".format(epoch,torch.tensor(losses).mean()))
