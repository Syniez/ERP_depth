import numpy as np
import torch




#def validation(model, val_loader, ):
#    with torch.no_grad():
#        for batch, (rgb, depth, mask) in tqdm(enumerate(val_loader)):
#            rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()
#            pred = Net(rgb)



def compute_errors(perd, gt, mask):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log, silog=silog, sq_rel=sq_rel)


def make_angle_maps():
    theta = torch.linspace(-pi, pi, 512).cuda()
    phi = torch.linspace(-pi/2, pi/2, 256).cuda()

    theta = np.tile(theta, reps=[256, 1])
    phi = np.tile(phi, reps=[512, 1]).T

    return theta, phi