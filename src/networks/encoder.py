from torch import nn

from networks import resnet
from utils import constants
from utils import view_extractor as ve


class DisjointEncoder(nn.Module):
    def __init__(self, num_classes, num_local, device):
        super(DisjointEncoder, self).__init__()
        self.num_classes = num_classes
        self.num_local = num_local
        self.extractor = resnet.resnet50(pretrained=True, pth_path=constants.PRETRAINED_EXTRACTOR_PATH)
        self.DEVICE = device

    def forward(self, x):
        global_view = ve.extract_global(x, self.extractor).to(self.DEVICE)
        global_fm, global_embed, _ = self.extractor(global_view.detach())

        local_views = ve.extract_local(global_view, self.num_local)
        _, local_embeds, _ = self.extractor(local_views)
        local_embeds = local_embeds.reshape(len(x), self.num_local, -1)

        return global_embed, local_embeds
