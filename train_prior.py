from typing import Sequence, List, Tuple
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import torch
from torch import Tensor
from torch.nn import MSELoss, PairwiseDistance
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from model.prior import NearestPointPredictor
from dataset.shapenet import AVAILABLE_CATEGORIES
from dataset.prior import PriorDataset
from utils.logging import get_predefined_logger, get_timestamp, LoggingConfig



logger = get_predefined_logger(__name__)
timestamp = get_timestamp()
LoggingConfig.set_level(logging.INFO)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def train(net, dataloader, loss_fn, optimizer, scheduler, writer, epoch_no):
    net.train()
    progress_bar = tqdm(dataloader, desc=f'Training ({epoch_no})', unit='batch')
    for i, data_item in enumerate(progress_bar):
        KNNs, nearest_points = data_item
        KNNs = KNNs.to(device)
        nearest_points = nearest_points.float().to(device)

        optimizer.zero_grad()

        predicted_points = net(KNNs)
        loss_value = loss_fn(predicted_points, nearest_points)
        loss_value = torch.mean(loss_value)
        loss_value.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar('Euclidean Loss (train)', loss_value, len(dataloader) * epoch_no + i)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], len(dataloader) * epoch_no + i)
        logger.info('batch {:>5}, train loss: {}'.format(i, loss_value.item()))

@torch.no_grad
def test(net, dataloader, loss_fn, writer, epoch_no) -> NDArray:
    logger.info('switch the prior network to evaluation mode')
    net.eval()
    loss_values = []
    progress_bar = tqdm(dataloader, desc=f'Testing ({epoch_no})', unit='batch')
    for i, data_item in enumerate(progress_bar):
        KNNs, nearest_points = data_item
        KNNs = KNNs.to(device)
        nearest_points = nearest_points.float().to(device)

        predicted_points = net(KNNs)

        loss_value = loss_fn(predicted_points, nearest_points)
        loss_value = torch.mean(loss_value)
        loss_values.append(loss_value.item())
        logger.info('batch {:>5}, test loss: {}'.format(i, loss_value.item()))
    loss_values = np.array(loss_values)
    return loss_values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', help='Path to the prior database file (stronly recommended to be stored in an SSD)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--max-epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('-K', type=int, default=100, help='Number of points in KNN')
    parser.add_argument('--n-query-points', type=int, default=4096, help='Number of query points per batch')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index')
    parser.add_argument('-C', '--checkpoint', default=None, help='Path to the checkpoint you want to load')
    config = parser.parse_args()
    LoggingConfig.set_file(f'log/prior/{timestamp}.log')
    logger.info('configurations:')
    for key, value in vars(config).items():
        logger.info(f'{key}={value}')

    global device
    device = torch.device(f'cuda:{config.gpu}')
    logger.info(f'use device {device} for training')

    net = NearestPointPredictor(config.K).to(device)
    logger.info(f'network architecture:\n{net}')
    if config.checkpoint is not None:
        ckpt_path = Path(config.checkpoint)
        state_dict = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state_dict)
    LoggingConfig.disable_console_output()

    db_path = Path(config.database)
    train_set = PriorDataset(db_path, True, K=config.K)
    test_set = PriorDataset(db_path, False, K=config.K)
    train_loader = DataLoader(train_set, batch_size=config.n_query_points, shuffle=False, num_workers=30)
    test_loader = DataLoader(test_set, batch_size=config.n_query_points, shuffle=False, num_workers=30)

    loss_fn = PairwiseDistance(eps=0)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    # Using a stepLR scheduler is faster and simpler, but the performance will be a little worse.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader), eta_min=config.learning_rate / 10)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4000, gamma=0.9)
    writer = SummaryWriter(log_dir='log/prior')

    for epoch_no in range(config.max_epochs):
        logger.info(f'epoch {epoch_no} starts')
        train(net, train_loader, loss_fn, optimizer, scheduler, writer, epoch_no)
        # Record the distribution of weights of the first layer
        for layer_name, parameters in net.named_parameters():
            if 'encoder' in layer_name:
                writer.add_histogram(f'{layer_name}/weights', parameters.cpu().detach().numpy(), epoch_no)
                writer.add_histogram(f'{layer_name}/grad', parameters.grad.cpu().detach().numpy(), epoch_no)
                break
        torch.save(net.state_dict(), f'pretrained-models/model-{epoch_no}.pth')
        logger.info(f'prior model weights (after epoch {epoch_no}) are saved')

        loss_values = test(net, test_loader, loss_fn, writer, epoch_no)
        writer.add_histogram('Euclidean distance (test)', loss_values, epoch_no)
        logger.info(f'epoch {epoch_no} ends')

    writer.flush()
    writer.close()
    logger.info('training ends successfully')
    logging.shutdown()


if __name__ == '__main__':
    main()
