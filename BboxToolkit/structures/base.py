import inspect
from abc import ABCMeta, abstractmethod


class BaseBbox(metaclass=ABCMeta):
    '''
    Base Bounding Box (BaseBbox): This class is the Base class for all
    types of Bboxes. In this class, we design some abstract functions
    which need to be implemented in subclasses.
    '''

    BBOX_CLASSES = dict() # Store Bbox classes
    TRAN_SHORTCUTS = dict() # Store transformation shortcut

    @classmethod
    def register_bbox_cls(cls, force=False):
        '''Register BaseBbox subclass.

        Args:
            force (bool): whether register the subclass when a same name class
            has been registered.

        Returns:
            Registrar.
        '''

        def  _decorator(sub_cls):
            assert issubclass(sub_cls, BaseBbox)
            cls_name = sub_cls.__name__.lower()

            if (not force) and (cls_name in cls.BBOX_CLASSES):
                raise KeyError(f'The {cls_name} class is already registered.')

            cls.BBOX_CLASSES[cls_name] = sub_cls
            return sub_cls
        return _decorator

    @classmethod
    def register_shortcuts(cls, start, end, force=False):
        '''Register functions as shortucts of transformation.

        Args:
            start (str (e.g., 'hbb')): function's input Bbox type name.
            end (str (e.g., 'obb')): function's output Bbox type name. Must
                different with 'start'.
            force (bool): whether register the shortcuts when a same name
                shortcut has been registered.

        Returns:
            Registrar.
        '''
        start, end = start.lower(), end.lower()
        assert start != end, 'The types of start and end are same.'
        assert start in cls.BBOX_CLASSES
        assert end in cls.BBOX_CLASSES
        key = start + '_2_' + end

        # To judge if the shortcuts has been registered.
        if (not force) and (key in cls.TRAN_SHORTCUTS):
            raise KeyError(f'The {key} shortcut is already registered.')

        def _decorator(func):
            cls.TRAN_SHORTCUTS[key] = func
            return func
        return _decorator

    def to_type(self, new_type):
        '''Transform Bboxes to another type. This funcution will firstly
           use registered shortcuts to transform Bboxes. Or, it will
           convert Bboxes using to_poly and run from_poly.

        Args:
            new_type (str (e.g., 'hbb')): the target type name of Bboxes.

        Returns:
            new_type: transformed Bboxes.
        '''
        new_type = new_type.lower()
        assert new_type in self.BBOX_CLASSES
        new_cls = self.BBOX_CLASSES[new_type]

        # Target type is same with now type, just output a copy of self.
        if isinstance(self, new_cls):
            return self.copy()

        # The shortcut has been registered, use shortcut to transform Bboxes.
        start_type = self.__class__.__name__.lower()
        key = start_type + '_2_' + new_type
        if key in self.TRAN_SHORTCUTS:
            return self.TRAN_SHORTCUTS[key](self)

        polys = self.to_poly()
        return new_cls.from_poly(polys)

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

    @classmethod
    @abstractmethod
    def concatenate(cls, bboxes):
        '''Concatenate list of bboxes.'''
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
