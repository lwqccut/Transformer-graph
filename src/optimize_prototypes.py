# Obtain hyperspherical prototypes prior to network training.

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

# Compute the loss related to the prototypes.
#
def prototype_loss(prototypes):
    # Dot product of normalized prototypes is cosine similarity.
    product = torch.matmul(prototypes, prototypes.t()) + 1
    # Remove diag from loss.
    product -= 2. * torch.diag(torch.diag(product))
    # Minimize maximum cosine similarity.
    loss = product.max(dim=1)[0]
    return loss.mean(), product.max()

#
# Main entry point of the script.
#
def create_hypersphere(num_classes, output_dimension, max_epoch=2000):

    prototypes = torch.randn(num_classes, output_dimension)
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1), requires_grad = True)
    optimizer = optim.SGD([prototypes], lr=.01, momentum=.9)

    # Optimize for separation.
    for epoch in range(max_epoch):
        # Compute loss.
        loss, sep = prototype_loss(prototypes)
        # Update.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("Epoch {0}: {1}".format(epoch, sep))
        # Renormalize prototypes.
        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1), requires_grad = True)
        optimizer = optim.SGD([prototypes], lr=.01, momentum=.9)

    return prototypes.detach()


if __name__ == '__main__':
    class_matched_points = create_hypersphere(3, 10, ['1', '2', '3'])
    print(class_matched_points)