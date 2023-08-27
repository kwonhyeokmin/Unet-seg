from config import cfg
import torch
from dataset.dataset import CTDataset
from dataset.uwmgi import UWMGIDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from models.unet import UNet
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import os.path as osp
import argparse


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=False, help='Path to pretrained checkpoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = make_args()

    # Select device (gpu | cpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ********************
    # 1. Load datasets
    # ********************
    dataset = UWMGIDataset()
    dataset_loader = CTDataset(dataset, is_train=True, transforms=cfg.data_transforms['train'])
    data_generator = DataLoader(dataset=dataset_loader, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=cfg.num_thread, pin_memory=True)

    # ****************
    # 2. Setting Loss function
    # ****************
    JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
    DiceLoss = smp.losses.DiceLoss(mode='multilabel')
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()
    LovaszLoss = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
    TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

    # ****************
    # 3. Training
    # ****************
    # load model
    model = UNet(in_channels=3, n_classes=3, n_channels=48).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    for epoch in range(1, cfg.num_epochs + 1):
        pbar = tqdm(enumerate(data_generator), total=len(data_generator), desc=f'Train - epoch: {epoch}')

        avg_loss = 0.0
        for step, (images, masks) in pbar:
            B = images.shape[0]
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            y_pred = model(images)
            losses = 0.5 * BCELoss(y_pred, masks) + 0.5 * TverskyLoss(y_pred, masks)

            # Gradient update
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            _losses = float(losses.detach().cpu().numpy())
            pbar.set_description('[epoch %d] Train - Loss %.04f' % (epoch, _losses))
            avg_loss += _losses

        pbar.set_description('[epoch %d] Train Loss - %.04f' % (epoch, avg_loss / B))

        # model save
        file_path = osp.join(cfg.model_dir, f'snapshot_{epoch}.pt')
        torch.save(model.state_dict(), file_path)

    # model save
    file_path = osp.join(cfg.model_dir, f'snapshot_{cfg.num_epochs + 1}.pt')
    torch.save(model.state_dict(), file_path)
    print('End training')
