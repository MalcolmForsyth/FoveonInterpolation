import torch
from metrics import PSNR, SSIM



class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    
crit = RMSELoss()   

class Evaluator():
    def __init__(self):
        self.PSNR = PSNR()
        self.SSIM = SSIM()
        self.RMSE = RMSELoss()
    def eval(self, pred, gt):
        rmse = self.RMSE(pred, gt)
        psnr = self.PSNR(pred, gt)
        ssim = self.SSIM(pred,gt)
        return {
                'rmse': rmse.item(),
               'psnr': psnr.item(),
               'ssim': ssim.item()
               }
