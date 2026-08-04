"""
Routines and objects for working with dlib's image dataset metadata XML files.
"""
from __future__ import annotations
import _dlib_pybind11
import collections.abc
import typing
__all__: list[str] = ['box', 'boxes', 'dataset', 'image', 'images', 'load_image_dataset_metadata', 'parts', 'save_image_dataset_metadata']
class box:
    """
    This object represents an annotated rectangular area of an image. 
    It is typically used to mark the location of an object such as a 
    person, car, etc.
    
    The main variable of interest is rect.  It gives the location of 
    the box.  All the other variables are optional.
    """
    class gender_type:
        """
        Members:
        
          MALE
        
          FEMALE
        
          UNKNOWN
        """
        FEMALE: typing.ClassVar[box.gender_type]  # value = <gender_type.FEMALE: 2>
        MALE: typing.ClassVar[box.gender_type]  # value = <gender_type.MALE: 1>
        UNKNOWN: typing.ClassVar[box.gender_type]  # value = <gender_type.UNKNOWN: 0>
        __members__: typing.ClassVar[dict[str, box.gender_type]]  # value = {'MALE': <gender_type.MALE: 1>, 'FEMALE': <gender_type.FEMALE: 2>, 'UNKNOWN': <gender_type.UNKNOWN: 0>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    FEMALE: typing.ClassVar[box.gender_type]  # value = <gender_type.FEMALE: 2>
    MALE: typing.ClassVar[box.gender_type]  # value = <gender_type.MALE: 1>
    UNKNOWN: typing.ClassVar[box.gender_type]  # value = <gender_type.UNKNOWN: 0>
    difficult: bool
    gender: ...
    ignore: bool
    label: str
    occluded: bool
    rect: _dlib_pybind11.rectangle
    truncated: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def age(self) -> float:
        ...
    @age.setter
    def age(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def angle(self) -> float:
        ...
    @angle.setter
    def angle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def detection_score(self) -> float:
        ...
    @detection_score.setter
    def detection_score(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def parts(self) -> parts:
        ...
    @property
    def pose(self) -> float:
        ...
    @pose.setter
    def pose(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class boxes:
    """
    An array of dlib::image_dataset_metadata::box objects.
    """
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, s: slice) -> boxes:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> box:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: boxes) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[box]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: box) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: boxes) -> None:
        """
        Assign list elements using a slice object
        """
    def __str__(self) -> str:
        ...
    def append(self, x: box) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def extend(self, L: boxes) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: box) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> box:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> box:
        """
        Remove and return the item at index ``i``
        """
class dataset:
    """
    This object represents a labeled set of images.  In particular, it contains the filename for each image as well as annotated boxes.
    """
    comment: str
    name: str
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def images(self) -> ...:
        ...
    @images.setter
    def images(self, arg0: ..., std: ...) -> None:
        ...
class image:
    """
    This object represents an annotated image.
    """
    filename: str
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def boxes(self) -> ...:
        ...
    @boxes.setter
    def boxes(self, arg0: ..., std: ...) -> None:
        ...
class images:
    """
    An array of dlib::image_dataset_metadata::image objects.
    """
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, s: slice) -> images:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> image:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: images) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[image]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: image) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: images) -> None:
        """
        Assign list elements using a slice object
        """
    def __str__(self) -> str:
        ...
    def append(self, x: image) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def extend(self, L: images) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: image) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> image:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> image:
        """
        Remove and return the item at index ``i``
        """
class parts:
    """
    This object is a dictionary mapping string names to object part locations.
    """
    def __bool__(self) -> bool:
        """
        Check whether the map is nonempty
        """
    def __delitem__(self, arg0: str) -> ...:
        ...
    def __getitem__(self, arg0: str) -> _dlib_pybind11.point:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: dict) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[str]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: str, arg1: _dlib_pybind11.point) -> None:
        ...
    def __str__(self) -> str:
        ...
    def items(self) -> collections.abc.Iterator[tuple[str, _dlib_pybind11.point]]:
        ...
def load_image_dataset_metadata(filename: str) -> dataset:
    """
    Attempts to interpret filename as a file containing XML formatted data as produced by the save_image_dataset_metadata() function.  The data is loaded and returned as a dlib.image_dataset_metadata.dataset object.
    """
def save_image_dataset_metadata(data: dataset, filename: str) -> None:
    """
    Writes the contents of the meta object to a file with the given filename.  The file will be in an XML format, although any extension can be used to name the file.
    """
