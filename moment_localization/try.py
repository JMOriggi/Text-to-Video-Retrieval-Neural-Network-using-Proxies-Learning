import torch
import torch.nn.functional as F

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

'''
a = torch.randn((16, 256, 512))
b = torch.randn((5046, 512))
b = torch.reshape(b, (1, b.shape[0], b.shape[1]))
b = b.repeat(a.shape[0], 1, 1)
print(f'A.shape : {a.shape}')
print(f'B.shape : {b.shape}')
print(f'a[0] : {a[0]}')
print(f'b : {b}')

#a = a.contiguous().view(32, -1, 512)
print(f'a.shape : {a.shape}')

def fun(x, y):
    #print(f'x.shape : {x.shape}')
    #print(f'y.shape : {y.shape}')
    I = torch.cat([x,y])
    #print(f'I.shape : {I.shape}')
    D = pairwise_distance(I, squared = True)[:x.size()[0], x.size()[0]:]
    return D

A = list(map(fun, torch.unbind(a, 0), b))
print(f'len A: {len(A)}')
print(f'len A[0]: {len(A[0])}')
print(f'len A[0].size: {A[0].shape}')
output = torch.stack(A, 0)
print(f'output.shape : {output.shape}')
print(f'output[0] : {output[0]}')


# Prepare Proxy mask
#T = binarize_mask(T, len(P))
#print(f'T.shape: {T.shape}')
#print(f'T: {T}')

# Compute loss
loss = torch.sum(- 1 * F.log_softmax(-output, -1), -1)
print(f'loss.shape: {loss.shape}')
print(f'loss: {loss}')
loss = torch.reshape(loss, (loss.shape[0], 16, 16))
print(f'loss.shape: {loss.shape}')
print(f'loss: {loss}')
'''

a = torch.randn((3, 3, 4))
print(f'a.shape : {a.shape}')
print(f'a[0] : {a}')

a = a[torch.triu(torch.ones(3, 3)) == 1]
print(f'a.shape : {a.shape}')
print(f'a[0] : {a}')

