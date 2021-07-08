import time


LOADING_FUNC = {} # Store loading functions of datasets
DUMPING_FUNC = {} # Store dumping functions of datasets

def register_io_func(load_or_dump, dataset_type, force=False):
    '''Register loading or dumping functions of a dataset.

    Args:
         load_or_dump (str): 'load' or 'dump'.
         dataset_type (str): dataset name (e.g. voc, coco).
         force (bool): forcely register functions when a function with
            same key has been register.

    Returns:
        decorator.
    '''
    assert load_or_dump in ['load', 'dump']
    func_dict = LOADING_FUNC if load_or_dump == 'load' \
            else DUMPING_FUNC

    dataset_type = dataset_type.lower()
    if (not force) and (dataset_type in func_dict):
        raise KeyError(f'{load_or_dump.catitalize()}ing function of ',
                       f'{dataset_type} is already registered!')

    def _decorator(func):
        func_dict[dataset_type] = func
    return _decorator


def load_dataset(dataset_type, loading_args):
    '''Select loading function of dataset_type and load dataset.

    Args:
        dataset_type (str | list[str]): dataset name used to select loading functions.
        loading_args (dict | list[str]): arguments of dataset_type loading functions.
        logger (logger): logger object.

    Return:
        output format:
        [
            {
                'filename': 'a.jpg',
                'width': 1024,
                'height': 2014,
                ... (image level informations),
                'ann': {
                    'bboxes': <obj:BaseBbox>,
                    'categories': <list[str]>,
                    ... (instance level informations)
                }
            },
            ...
        ]
    '''
    if not isinstance(dataset_type, list):
        dataset_type = [dataset_type]
    if not isinstance(loading_args, list):
        loading_args = [loading_args]
    assert len(dataset_type) == len(loading_args)

    start_time = time.time()
    data = []
    for i, (dtype, largs) in enumerate(zip(dataset_type, loading_args)):
        print(f'#### Num{i} Dataset_type: {dtype}.')
        for k, v in loading_args.items():
            print(f'{k}: {v}')

        if dtype not in LOADING_FUNC:
            raise KeyError(f'No loading function of {dtype}')
        loading_func = LOADING_FUNC[dataset_type.lower()]
        data.extend(loading_func(**largs))
    end_time = time.time()

    print(f'Time consuming: {end_time - start_time:.3f}s.')
    print(f'Data number: {len(data)}')
    return data


def dump_dataset(dataset_type, dumping_args):
    '''Select loading function of dataset_type and load dataset.

    Args:
        dataset_type (str): dataset name used to select dumping functions.
        loading_args (dict): arguments of dataset_type dumping functions.
        logger (logger): logger object.

    Return:
        None
    '''
    if not isinstance(dataset_type, list):
        dataset_type = [dataset_type]
    if not isinstance(dumping_args, list):
        dumping_args = [dumping_args]

    start_time = time.time()
    for i, (dtype, dargs) in enumerate(zip(dataset_type, dumping_args)):
        print(f'#### Num{i} Dataset_type: {dtype}.')
        for k, v in dumping_args.items():
            print(f'{k}: {v}')

        if dataset_type not in DUMPING_FUNC:
            raise KeyError(f'No dumping function of {dataset_type}')
        dumping_func = DUMPING_FUNC[dataset_type.lower()]
        dumping_func(**dargs)
    end_time = time.time()

    print(f'Time consuming: {end_time - start_time:.3f}s.')
