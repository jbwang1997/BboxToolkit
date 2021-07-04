'''Base functions for loading and dumping datasets.'''
import time


# Dictionarys store loading and dumping functions
LOADING_FUNC = {}
DUMPING_FUNC = {}

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
        dataset_type (str): dataset name used to select loading functions.
        loading_args (dict): arguments of dataset_type loading functions.
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
                    'classes': <list[str]>,
                    ... (instance level informations)
                }
            },
            ...
        ]
    '''
    if dataset_type not in LOADING_FUNC:
        raise KeyError(f'No loading function of {dataset_type}')
    loading_func = LOADING_FUNC[dataset_type.lower()]

    print(f'Start loading {dataset_type}.')
    for k, v in loading_args.items():
        print(f'{k}: {v}')

    start_time = time.time()
    data = loading_func(**loading_args)
    end_time = time.time()

    print(f'Finish loading {dataset_type}.')
    print(f'Time consuming: {end_time - start_time:.3f}s.')
    print(f'Data number: {len(data)}')
    return data


def dump_dataset(dataset_type, dumping_dict):
    '''Select loading function of dataset_type and load dataset.

    Args:
        dataset_type (str): dataset name used to select dumping functions.
        loading_args (dict): arguments of dataset_type dumping functions.
        logger (logger): logger object.

    Return:
        None
    '''
    if dataset_type not in DUMPING_FUNC:
        raise KeyError(f'No dumping function of {dataset_type}')
    dumping_func = DUMPING_FUNC[dataset_type.lower()]

    print(f'Start dumping data into {dataset_type} format.')
    for k, v in dumping_dict.items():
        print(f'{k}: {v}')

    start_time = time.time()
    dumping_func(**dumping_dict)
    end_time = time.time()

    print(f'Finish dumping data into {dataset_type} format.')
    print(f'Time consuming: {end_time - start_time:.3f}s.')
