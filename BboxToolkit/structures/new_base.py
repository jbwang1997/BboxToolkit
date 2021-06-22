from abc import ABCMeta, abstractmethod


class BaseBbox(metaclass=ABCMeta):
    '''
    This class is the Base class for all types of Bboxes. In this class,
    we design the feature of transformation and some abstract functions.
    '''

    # A dictionary contain shortcuts of transformation.
    TRAN_SHORTCUTS = dict()

    @classmethod
    def register_shortcuts(cls, start, end, force=False):
        '''Register functions as shortucts of transformation.

        Args:
            start (BaseBbox subclass (e.g., HBB)): functions input Bbox type.
            end (BbaseBox subclass (e.g., OBB)): functions output Bbox type.
            force (bool): whether register the shortcuts when a same name
                shortcut has been registered.

        Returns:
            Registrar.
        '''
        assert isinstance(start, BaseBbox)
        assert isinstance(end, BaseBbox)
        assert start is not end, 'The types of start and end are same.'
        key = start.__name__ + '2' + end.__name__

        # To judge if the shortcuts has been registered.
        if (not force) and (key in cls.TRAN_SHORTCUTS):
            raise KeyError(f'The shortcut {key} is already registered.')

        def _decorator(func):
            cls.TRAN_SHORTCUTS[key] = func
            return func
        return _decorator

    def to_type(self, new_type):
        '''Transform Bboxes to another type. This funcution will firstly
           use registered shortcuts to transform Bboxes. Or, it will
           convert Bboxes using to_poly and run from_poly.

        Args:
            new_type (BboxToolkit.strutures): the target type of Bboxes.

        Returns:
            new_type: transformed Bboxes.
        '''
        assert isinstance(new_type, BaseBbox)

        # Target type is same with now type, just output a copy of self.
        if isinstance(self, new_type):
            return self.copy()

        # The shortcut has been registered, use shortcut to transform Bboxes.
        key = type(self).__name__ + '2' + new_type.__name__
        if key in self.TRAN_SHORTCUTS:
            return self.TRAN_SHORTCUTS[key](self)

        polys = self.to_poly()
        return new_type.from_poly(polys)

    @abstractmethod
    def __iter__(self):
        '''Iterate all Bboxes in polygon form.'''
        pass

    @abstractmethod
    def __getitem__(self, index):
        '''Index the Bboxes

        Args:
            index (int | ndarray): Indices in the format of interger or ndarray.

        Returns:
            type(self): indexed Bboxes.
        '''
        pass

    @abstractmethod
    def __len__(self):
        '''Number of Bboxes.'''
        pass

    @abstractmethod
    def to_poly(self):
        '''Output the Bboxes polygons (list[list[np.ndarry]]).'''
        pass

    @abstractmethod
    @classmethod
    def from_poly(cls, polys):
        '''Create a Bbox instance from polygons (list[list[np.ndarray]]).'''
        pass

    @abstractmethod
    def copy(self):
        '''Copy this instance.'''
        pass

    @abstractmethod
    def gen_empty(self):
        '''Create a Bbox instance with len == 0.'''
        pass

    @abstractmethod
    def areas(self):
        '''ndarry: areas of each instance.'''
        pass

    @abstractmethod
    def warp(self, M):
        '''Warp the Bboxes.

        Args:
            M (ndarray): 2x3 or 3x3 matrix.

        Returns:
            Warped Bboxes.
        '''
        pass

    @abstractmethod
    def flip(self, W, H, direction='horizontal'):
        '''Flip Bboxes alone the given direction.

        Args:
            W (int | float): image width.
            H (int | float): image height.
            direction (str): 'horizontal' or 'vertical' or 'digonal'.

        Returns:
            type(self): The flipped masks.
        '''
        pass

    @abstractmethod
    def translate(self, x, y):
        '''Translate the Bboxes.

        Args:
            x (int | float): translation along x axis.
            y (int | float): translation along y axis.

        Returns:
            Translated Bboxes.
        '''
        pass

    @abstractmethod
    def resize(self, ratios):
        '''Resize Bboxes according the ratios.

        Args:
            ratios (int | float | tuple): the resize ratios.

        Returns:
            Resized Bboxes.
        '''
        pass
