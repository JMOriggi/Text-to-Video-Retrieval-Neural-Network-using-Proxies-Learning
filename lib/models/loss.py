import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import math

# COMMON LOSS
def bce_rescale_loss(scores, masks, targets, cfg):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    joint_prob = torch.sigmoid(scores) * masks
    target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    loss = F.binary_cross_entropy(joint_prob, target_prob, reduction='none') * masks
    loss_value = torch.sum(loss) / torch.sum(masks)
    return loss_value, joint_prob

# PROXY LOSS
def binarize_mask(T, nb_classes):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes = range(0,nb_classes))
    T = torch.FloatTensor(T).cuda()
    return T
def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes = range(0, nb_classes))
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()
    return T
def pairwise_distance(a, squared=False):
    """Computes the pairwise distance matrix with numerical stability."""
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)) - 2 * (torch.mm(a, torch.t(a)))
    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)
    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)
    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)
    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(pairwise_distances,(error_mask == False).float())
    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(),device=pairwise_distances.device)
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

class ProxyNCA_prob(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale, config, **kwargs):
        print(f'INIT PROXY')
        torch.nn.Module.__init__(self)
        self.config = config
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale
        self.nb_classes = nb_classes
        print(f'scale={self.scale}')
        print(f'proxy={self.proxies}')
        print(f'size proxy={self.nb_classes}x{sz_embed}')
        
    def forward(self, X, T, masks):
        #print('------- FORWARD PROXY')
        
        # Get Proxies
        P = self.proxies.cuda()
        
        # Normalize
        P = self.scale * F.normalize(P, p = 2, dim = -1) #L_p norm of the input on the embedding
        X = self.scale * F.normalize(X, p = 2, dim = -1)
        
        # Reshape Proxies and X
        P = torch.reshape(P, (1, P.shape[0], P.shape[1]))
        P = P.repeat(X.shape[0], 1, 1)
        X = X.permute(0, 3, 2, 1)
        X = torch.flatten(X, start_dim=1,end_dim=2)
        #print(f'X.shape : {X.shape}')
        #print(f'P.shape: {P.shape}')
        
        # Compute similarity
        def fun(x, y):
            I = torch.cat([x,y])
            D = pairwise_distance(I, squared = True)[:x.size()[0], x.size()[0]:]
            return D
        D = list(map(fun, torch.unbind(X, 0), P))
        D = torch.stack(D, 0)
        #print(f'D.shape : {D.shape}')
        #rint(f'D[0]: {D[0]}')
        
        # Prepare Proxy labels mask
        T = binarize_mask(T, self.nb_classes)
        T = T.unsqueeze(1)
        T = T.repeat(1, X.shape[1], 1)
        #print(f'T.shape: {T.shape}')
        #print(f'T[0]: {T[0]}')
        
        # Compute loss (normal NCApp ends here)
        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        #print(f'loss.shape: {loss.shape}')
        #print(f'loss[0]: {loss[0]}')
        
        # Normalize and mask
        loss_scores = torch.reshape(loss, (loss.shape[0], 1, 16, 16))
        loss_scores = loss_scores * masks
        '''
        loss_scores = loss_scores.squeeze(1)
        print(f'loss_scores.shape: {loss_scores.shape}')
        print(f'loss_scores[0]: {loss_scores[0]}')
        def norm(AA):
            batch_size = AA.shape[0]
            height = 16
            width = 16
            AA = AA.view(AA.size(0), -1)
            AA -= AA.min(1, keepdim=True)[0]
            AA /= AA.max(1, keepdim=True)[0]
            AA = AA.view(batch_size, height, width)
            return AA
        print(f'loss_scores.shape: {loss_scores.shape}')
        loss_scores = norm(loss_scores)
        loss_scores = torch.reshape(loss, (loss.shape[0], 1, 16, 16))
        print(f'loss_scores.shape norm: {loss_scores.shape}')
        print(f'loss_scores[0] norm: {loss_scores[0]}')
        '''
        
        # Average Loss with mask
        loss_value = torch.sum(loss_scores) / torch.sum(masks)
        loss_value = loss_value/100
        #print(f'loss_value: {loss_value}')
        #print(f'masks.shape: {masks.shape}')
        #print(f'masks[0]: {masks[0]}')
        
        return loss_value, loss_scores
    
    
    def forward_single(self, X, T):
        #print('------- FORWARD PROXY')
        
        # Get Proxies
        P = self.proxies.cuda()
        
        # Normalize
        P = self.scale * F.normalize(P, p = 2, dim = -1) #L_p norm of the input on the embedding
        X = self.scale * F.normalize(X, p = 2, dim = -1)
        
        # Reshape Proxies to fit the model output
        X = torch.flatten(X, start_dim=1)
        #print(f'P.shape: {P.shape}')
        #print(f'X.shape : {X.shape}')
        
        # Prepare for distances computation
        A = torch.cat([X, P])
        #print(f'A.shape: {A.shape}')
        
        # Compute distances
        D = pairwise_distance(A, squared = True)[:X.size()[0], X.size()[0]:]
        #print(f'D.shape : {D.shape}')
        #print(f'D : {D}')
        
        # Prepare Proxy mask
        T = binarize_mask(T, self.nb_classes)
        #print(f'T.shape: {T.shape}')
        #print(f'T: {T}')

        # Compute loss
        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()/100
        
        return loss

'''        
    def forward(self, X, T, masks):
        print('------- FORWARD PROXY')
        
        # Get Proxies
        P = self.proxies.cuda()
        
        # Normalize
        P = self.scale * F.normalize(P, p = 2, dim = -1) #L_p norm of the input on the embedding
        X = self.scale * F.normalize(X, p = 2, dim = -1)
        
        # Reshape Proxies and X
        P = torch.reshape(P, (1, P.shape[0], P.shape[1]))
        P = P.repeat(X.shape[0], 1, 1)
        X = X.permute(0, 3, 2, 1)
        X = torch.flatten(X, start_dim=1,end_dim=2)
        print(f'X.shape : {X.shape}')
        print(f'P.shape: {P.shape}')
        
        # Compute similarity
        def fun(x, y):
            I = torch.cat([x,y])
            D = pairwise_distance(I, squared = True)[:x.size()[0], x.size()[0]:]
            return D
        D = list(map(fun, torch.unbind(X, 0), P))
        D = torch.stack(D, 0)
        print(f'D.shape : {D.shape}')
        print(f'D[0]: {D[0]}')
        
        # Prepare Proxy labels mask
        T = binarize_mask(T, self.nb_classes)
        print(f'T.shape: {T.shape}')
        print(f'T: {T}')
        T = T.unsqueeze(1)
        T = T.repeat(1, X.shape[1], 1)
        print(f'T.shape: {T.shape}')
        print(f'T[0]: {T[0]}')
        
        # Compute loss (normal NCApp ends here)
        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        print(f'loss.shape: {loss.shape}')
        print(f'loss[0]: {loss[0]}')
        
        # Normalize
        loss = torch.sigmoid(loss)
        print(f'loss.shape: {loss.shape}')
        print(f'loss[0]: {loss[0]}')
        loss_scores = torch.reshape(loss, (loss.shape[0], 16, 16))
        print(f'loss.shape: {loss.shape}')
        print(f'loss_scores[0]: {loss_scores[0]}')
        #loss = loss.mean()
        #print(f'loss proxies: {loss}')
        
        # Average Loss with mask
        loss_scores = loss_scores * masks
        print(f'masks.shape: {masks.shape}')
        print(f'masks[0]: {masks[0]}')
        print(f'loss.shape: {loss.shape}')
        print(f'loss_scores[0]: {loss_scores[0]}')
        loss_value = torch.sum(loss_scores) / torch.sum(masks)
        print(f'loss_value.shape: {loss_value.shape}')
        print(f'loss_value: {loss_value}')
        
        return loss_value, loss_scores
    
    def forward(self, X, T, map_mask):
        #print('------- FORWARD PROXY')
        
        # Get Proxies
        P = self.proxies.cuda()
        
        # Normalize
        P = self.scale * F.normalize(P, p = 2, dim = -1) #L_p norm of the input on the embedding
        X = self.scale * F.normalize(X, p = 2, dim = -1)
        
        # Reshape Proxies to fit the model output
        P = torch.reshape(P, (P.shape[0], 1 ,1, P.shape[1]))
        P = P.repeat(1, 16, 16, 1)
        print(f'P.shape: {P.shape}')
        #X = X.permute(0, 3, 2, 1)
        print(f'X.shape : {X.shape}')
        
        
        # Prepare for distances computation
        A = torch.cat([X, P])
        print(f'A.shape: {A.shape}')
        
        # Compute distances
        D = pairwise_distance(A, squared = True)[:X.size()[0], X.size()[0]:]
        print(f'D.shape : {D.shape}')
        #print(f'D : {D}')
        
        # Prepare Proxy mask
        T = binarize_mask(T, len(P))
        print(f'T.shape: {T.shape}')
        #print(f'T: {T}')

        # Compute loss
        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        #print(f'loss: {loss}')
        loss = loss.mean()
        
        return loss
    '''