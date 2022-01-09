import numpy as np

def GenerateLocalEpochs(percentage, size, max_epochs):
    ''' Method generates list of epochs for selected clients
    to replicate system heteroggeneity

    Params:
      percentage: percentage of clients to have fewer than E epochs
      size:       total size of the list
      max_epochs: maximum value for local epochs

    Returns:
      List of size epochs for each Client Update

    '''

    # if percentage is 0 then each client runs for E epochs
    if percentage == 0:
        return np.array([max_epochs] * size)
    else:
        # get the number of clients to have fewer than E epochs
        heterogenous_size = int((percentage / 100) * size)

        # generate random uniform epochs of heterogenous size between 1 and E
        epoch_list = np.random.randint(1, max_epochs, heterogenous_size)

        # the rest of the clients will have E epochs
        remaining_size = size - heterogenous_size
        rem_list = [max_epochs] * remaining_size

        epoch_list = np.append(epoch_list, rem_list, axis=0)

        # shuffle the list and return
        np.random.shuffle(epoch_list)

        return epoch_list

