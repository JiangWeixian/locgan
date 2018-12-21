import torch
from torch.autograd import Variable
from modules.roi_pooling.modules.roi_pool import RoIPool

roi_7 = RoIPool(7, 7, 1.0)
fm = Variable(torch.randn(3, 3, 20, 20).cuda())
bboxes = Variable(torch.cuda.FloatTensor([
    [[1, 0, 0, 3, 3],[1, 0, 0, 3, 3]],
    [[1, 0, 0, 3, 3],[1, 0, 0, 3, 3]],
    [[1, 0, 0, 3, 3],[1, 0, 0, 3, 3]]
    ])
    )
roi_fm = roi_7(fm, bboxes)
print(roi_fm)