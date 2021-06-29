from abc import ABCMeta, abstractmethod


class BaseBbox(metaclass=ABCMeta):
    '''
    Base Bounding Box (BaseBbox): This class is the Base class for all
    types of Bboxes. In this class, we design some abstract functions
    which need to be implemented in subclasses.
    '''

    # A dictionary contain shortcuts of transformation.
    TRAN_SHORTCUTS = dict()

    @classmethod
    def register_shortcuts(cls, start, end, force=False):
        '''Register functions as shortucts of transformation.

        Args:
            start (BaseBbox subclass (e.g., HBB)): functions input Bbox type.
            end (BbaseBbox subclass (e.g., OBB)): functions output Bbox type.
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

    def __iter__(self):
        '''Iterate all Bboxes in polygon form.'''
        return iter(self.to_poly())

    @abstractmethod
    def __getitem__(self, index):
        '''Index the Bboxes

        Args:
            index (list | ndarray): Indices in the format of interger or ndarray.

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
        '''Output the Bboxes polygons (list[list[np.ndarry]]). The first level
        of the list corresponds to objects, the second level to the polys that
        compose the object, the third level to the poly coordinates.
        '''
        pass

    @classmethod
    @abstractmethod
    def from_poly(cls, polys):
        '''Create a Bbox instance from polygons (list[list[np.ndarray]]). The
        first level of the list corresonds to objects, the second level to the
        polys that compose the object, the third level to the poly coordinates.
        '''
        pass

    @abstractmethod
    def copy(self):
        '''Copy this instance.'''
        pass

    @classmethod
    @abstractmethod
    def gen_empty(cls):
        '''Create a Bbox instance with len == 0.'''
        pass

    @abstractmethod
    def areas(self):
        '''ndarry: areas of each instance.'''
        pass

    @abstractmethod
    def rotate(self, x, y, angle, keep_btype=True):
        '''Rotate the Bboxes.

        Args:
            x (int | float): x coordinate of rotating center.
            y (int | float): y coordinate of rotating center.
            angle (int | float): roatation angle.
            keep_btype (bool): if True, returned Bboxes type will be
                same with self, else return POLY. Default:True.

        Returns:
            Warped POLY
        '''
        pass

    @abstractmethod
    def warp(self, M, keep_btype=False):
        '''Warp the Bboxes.

        Args:
            M (ndarray): 2x3 or 3x3 matrix.
            keep_btype (bool): if True, returned Bboxes type will be
                same with self, else return POLY. Default:False.

        Returns:
            Warped POLY
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
            ratios (int | float | list | tuple): the resize ratios.

        Returns:
            Resized Bboxes.
        '''
        pass
