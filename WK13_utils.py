import torch

from IPython.display import display

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid as tv_make_grid

from data_utils import LFWUtils as LFWUtils_Linear
from image_utils import make_image


class LFWUtils(LFWUtils_Linear):
  @staticmethod
  def train_test_split(test_pct=0.5, random_state=101010):
    train, test = LFWUtils_Linear.train_test_split(test_pct=test_pct, random_state=random_state)
    x_train = Tensor(train["pixels"])
    y_train = Tensor(train["labels"]).long()

    x_test = Tensor(test["pixels"])
    y_test = Tensor(test["labels"]).long()

    return x_train, x_test, y_train, y_test


  @staticmethod
  def get_labels(model, dataloader):
    model.eval()
    with torch.no_grad():
      data_labels = []
      pred_labels = []
      for x, y in dataloader:
        y_pred = model(x).argmax(dim=1)
        data_labels += [l.item() for l in y]
        pred_labels += [l.item() for l in y_pred]
      return data_labels, pred_labels

  @staticmethod
  def count_parameters(model):
    intp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{intp:,}"

  @staticmethod
  def map01(x):
    return (x - x.min()) / (x.max() - x.min())

  @staticmethod
  def display_activation_grids(activations, idx, max_imgs=64):
    for layer,actv in activations.items():
      batch = actv.unsqueeze(2)
      grid = (255 * tv_make_grid(batch[idx, :max_imgs]))[0]
      print("")
      display(layer)
      display(make_image(grid, width=grid.shape[1]).resize((4*LFWUtils.IMAGE_SIZE[0], 4*LFWUtils.IMAGE_SIZE[1])))

  @staticmethod
  def display_kernel_grids(layer_kernels, max_imgs=64):
    for layer,kernel in layer_kernels.items():
      batch = kernel[:max_imgs, :3].abs()
      batch_img_mins = batch.min(dim=0, keepdim=True)[0]
      batch_img_maxs = batch.max(dim=0, keepdim=True)[0]
      batch = (batch - batch_img_mins) / (batch_img_maxs - batch_img_mins)
      grid = (255 * tv_make_grid(batch))
      print("")
      display(layer)
      display(make_image(grid.permute(1,2,0), width=grid.shape[1]).resize((256, 256)))
