def set_config(args):
    """ Model, Data, and Training Coniguration

    Specifies detailed configurations, such as batch-size, number of epcohs and rounds, 
    hyper-parameters, etc., can be set in this file.
    """
    #
    # scenarios
    if 'FM' in args.task:  
        # FedMatch data spilt setting
        args.useFedmatchDataLoader = True
        args.bsize_s = 256
        args.test_batchsize = 500
        args.dataset_id_to_name = {0: 'cifar_10'}
        args.dataset_id = 0
        args.num_clients = 100

    if 'FedProx' in args.method:  
        # FedMatch data spilt setting
        args.percentage = 50  # chose from 0, 50, 90
        args.local_epochs = 10

    if 'SL' not in args.method:
        args.use_PULoss = True

    return args


