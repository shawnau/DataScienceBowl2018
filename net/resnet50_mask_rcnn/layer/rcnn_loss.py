from common import *


# rcnn_loss uses deltas_sigma=1
def rcnn_loss(logits, deltas, labels, targets, deltas_sigma=1.0):
    batch_size, num_classes = logits.size(0), logits.size(1)
    # label_weights = Variable(torch.ones((batch_size))).cuda()
    # rcnn_cls_loss = weighted_cross_entropy_with_logits(logits, labels, label_weights)
    rcnn_cls_loss = F.cross_entropy(logits, labels, size_average=True)

    num_pos = len(labels.nonzero())
    if num_pos > 0:
        # one hot encode
        select = Variable(torch.zeros((batch_size,num_classes))).cuda()
        select.scatter_(1, labels.view(-1,1), 1)
        select[:,0] = 0
        select = select.view(batch_size,num_classes,1).expand((batch_size,num_classes,4)).contiguous().byte()

        deltas = deltas.view(batch_size,num_classes,4)
        deltas = deltas[select].view(-1,4)

        deltas_sigma2 = deltas_sigma*deltas_sigma
        rcnn_reg_loss = F.smooth_l1_loss(deltas*deltas_sigma2, targets*deltas_sigma2, size_average=False)/deltas_sigma2/num_pos
    else:
        rcnn_reg_loss = Variable(torch.cuda.FloatTensor(1).zero_()).sum()

    return rcnn_cls_loss, rcnn_reg_loss
