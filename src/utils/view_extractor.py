import torch
import torch.nn.functional as F
import torchvision.transforms as T
from skimage import measure


def extract_global(x, extractor):
    """
    Adopted from MMAL --
    Paper: https://arxiv.org/pdf/2003.09150.pdf
    Code: https://github.com/ZF4444/MMAL-Net
    """
    fms, _, fm1 = extractor(x)
    batch_size, channel_size, side_size, _ = fms.shape
    fm1 = fm1.detach()

    A = torch.sum(fms, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > a).float()

    A1 = torch.sum(fm1, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    M1 = (A1 > a1).float()

    coordinates = []
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(14, 14)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))

        intersection = ((component_labels == (max_idx + 1)).astype(int) + (M1[i][0].cpu().numpy() == 1).astype(
            int)) == 2
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            # print('there is one img no intersection')
        else:
            bbox = prop[0].bbox

        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)

    # SCDA
    global_view = torch.zeros([batch_size, 3, 448, 448])  # [N, 3, 448, 448]
    for i in range(batch_size):
        [x0, y0, x1, y1] = coordinates[i]
        global_view[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(448, 448),
                                             mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
    return global_view


def extract_local(global_view, num_local):
    random_cropper = T.Compose([T.RandomCrop(size=global_view.shape[-1] // 3), T.Resize((224, 224))])
    local_views = [random_cropper(global_view) for _ in range(num_local - 5)]
    local_views.extend(T.FiveCrop((224, 224))(global_view))

    local_views = torch.cat(local_views)
    return local_views
