from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
from . import cuda
from . import image_dataset_metadata
__all__: list[str] = ['DLIB_USE_BLAS', 'DLIB_USE_CUDA', 'DLIB_USE_LAPACK', 'KBD_MOD_ALT', 'KBD_MOD_CAPS_LOCK', 'KBD_MOD_CONTROL', 'KBD_MOD_META', 'KBD_MOD_NONE', 'KBD_MOD_NUM_LOCK', 'KBD_MOD_SCROLL_LOCK', 'KBD_MOD_SHIFT', 'KEY_ALT', 'KEY_BACKSPACE', 'KEY_CAPS_LOCK', 'KEY_CTRL', 'KEY_DELETE', 'KEY_DOWN', 'KEY_END', 'KEY_ESC', 'KEY_F1', 'KEY_F10', 'KEY_F11', 'KEY_F12', 'KEY_F2', 'KEY_F3', 'KEY_F4', 'KEY_F5', 'KEY_F6', 'KEY_F7', 'KEY_F8', 'KEY_F9', 'KEY_HOME', 'KEY_INSERT', 'KEY_LEFT', 'KEY_PAGE_DOWN', 'KEY_PAGE_UP', 'KEY_PAUSE', 'KEY_RIGHT', 'KEY_SCROLL_LOCK', 'KEY_SHIFT', 'KEY_UP', 'USE_AVX_INSTRUCTIONS', 'USE_NEON_INSTRUCTIONS', 'angle_between_lines', 'apply_cca_transform', 'array', 'as_grayscale', 'assignment_cost', 'auto_train_rbf_classifier', 'bottom_up_clustering', 'cca', 'cca_outputs', 'center', 'centered_rect', 'centered_rects', 'chinese_whispers', 'chinese_whispers_clustering', 'chip_details', 'chip_detailss', 'chip_dims', 'cnn_face_detection_model_v1', 'convert_image', 'convert_image_scaled', 'correlation_tracker', 'count_points_between_lines', 'count_points_on_side_of_line', 'count_steps_without_decrease', 'count_steps_without_decrease_robust', 'cross_validate_ranking_trainer', 'cross_validate_sequence_segmenter', 'cross_validate_trainer', 'cross_validate_trainer_threaded', 'cuda', 'distance_to_line', 'dot', 'dpoint', 'dpoints', 'drectangle', 'equalize_histogram', 'extract_image_4points', 'extract_image_chip', 'extract_image_chips', 'face_recognition_model_v1', 'fhog_object_detector', 'find_bright_keypoints', 'find_bright_lines', 'find_candidate_object_locations', 'find_dark_keypoints', 'find_dark_lines', 'find_line_endpoints', 'find_max_global', 'find_min_global', 'find_optimal_momentum_filter', 'find_optimal_rect_filter', 'find_peaks', 'find_projective_transform', 'full_object_detection', 'full_object_detections', 'function_evaluation', 'function_evaluation_request', 'function_spec', 'gaussian_blur', 'get_face_chip', 'get_face_chip_details', 'get_face_chips', 'get_frontal_face_detector', 'get_histogram', 'get_rect', 'global_function_search', 'grow_rect', 'hit_enter_to_continue', 'hough_transform', 'hysteresis_threshold', 'image_dataset_metadata', 'image_gradients', 'image_window', 'insert_image_chip', 'intersect', 'inv', 'jet', 'jitter_image', 'keyboard_mod_keys', 'label_connected_blobs', 'label_connected_blobs_watershed', 'length', 'line', 'load_grayscale_image', 'load_libsvm_formatted_data', 'load_rgb_alpha_image', 'load_rgb_image', 'make_bounding_box_regression_training_data', 'make_sparse_vector', 'matrix', 'max_cost_assignment', 'max_index_plus_one', 'max_point', 'max_point_interpolated', 'min_barrier_distance', 'mmod_rectangle', 'mmod_rectangles', 'mmod_rectangless', 'momentum_filter', 'no_convex_quadrilateral', 'non_printable_keyboard_keys', 'normalize_image_gradients', 'num_separable_filters', 'pair', 'partition_pixels', 'point', 'point_transform_projective', 'points', 'polygon_area', 'probability_that_sequence_is_increasing', 'pyramid_down', 'randomly_color_image', 'range', 'ranges', 'rangess', 'ranking_pair', 'ranking_pairs', 'rect_filter', 'rectangle', 'rectangles', 'rectangless', 'reduce', 'remove_incoherent_edge_pixels', 'resize_image', 'reverse', 'rgb_pixel', 'rvm_trainer_histogram_intersection', 'rvm_trainer_linear', 'rvm_trainer_radial_basis', 'rvm_trainer_sparse_histogram_intersection', 'rvm_trainer_sparse_linear', 'rvm_trainer_sparse_radial_basis', 'save_face_chip', 'save_face_chips', 'save_image', 'save_libsvm_formatted_data', 'scale_rect', 'segmenter_params', 'segmenter_test', 'segmenter_type', 'set_dnn_prefer_smallest_algorithms', 'shape_predictor', 'shape_predictor_training_options', 'shrink_rect', 'signed_distance_to_line', 'simple_object_detector', 'simple_object_detector_training_options', 'simple_test_results', 'skeleton', 'sobel_edge_detector', 'solve_structural_svm_problem', 'sparse_ranking_pair', 'sparse_ranking_pairs', 'sparse_vector', 'sparse_vectors', 'sparse_vectorss', 'spatially_filter_image', 'spatially_filter_image_separable', 'sub_image', 'suppress_non_maximum_edges', 'svm_c_trainer_histogram_intersection', 'svm_c_trainer_linear', 'svm_c_trainer_radial_basis', 'svm_c_trainer_sparse_histogram_intersection', 'svm_c_trainer_sparse_linear', 'svm_c_trainer_sparse_radial_basis', 'svm_rank_trainer', 'svm_rank_trainer_sparse', 'test_binary_decision_function', 'test_ranking_function', 'test_regression_function', 'test_sequence_segmenter', 'test_shape_predictor', 'test_simple_object_detector', 'threshold_filter_singular_values', 'threshold_image', 'tile_images', 'train_sequence_segmenter', 'train_shape_predictor', 'train_simple_object_detector', 'transform_image', 'translate_rect', 'vector', 'vectors', 'vectorss', 'zero_border_pixels']
class _binary_test:
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def class1_accuracy(self) -> float:
        """
        A value between 0 and 1, measures accuracy on the +1 class.
        """
    @class1_accuracy.setter
    def class1_accuracy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def class2_accuracy(self) -> float:
        """
        A value between 0 and 1, measures accuracy on the -1 class.
        """
    @class2_accuracy.setter
    def class2_accuracy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class _decision_function_histogram_intersection:
    def __call__(self, arg0: vector) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def alpha(self) -> vector:
        ...
    @property
    def b(self) -> float:
        ...
    @property
    def basis_vectors(self) -> vectors:
        ...
    @property
    def kernel_function(self) -> ...:
        ...
class _decision_function_linear:
    def __call__(self, arg0: vector) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def bias(self) -> float:
        ...
    @bias.setter
    def bias(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def weights(self) -> vector:
        ...
class _decision_function_polynomial:
    def __call__(self, arg0: vector) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def alpha(self) -> vector:
        ...
    @property
    def b(self) -> float:
        ...
    @property
    def basis_vectors(self) -> vectors:
        ...
    @property
    def kernel_function(self) -> ...:
        ...
class _decision_function_radial_basis:
    def __call__(self, arg0: vector) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def alpha(self) -> vector:
        ...
    @property
    def b(self) -> float:
        ...
    @property
    def basis_vectors(self) -> vectors:
        ...
    @property
    def kernel_function(self) -> _radial_basis_kernel:
        ...
class _decision_function_sigmoid:
    def __call__(self, arg0: vector) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def alpha(self) -> vector:
        ...
    @property
    def b(self) -> float:
        ...
    @property
    def basis_vectors(self) -> vectors:
        ...
    @property
    def kernel_function(self) -> ...:
        ...
class _decision_function_sparse_histogram_intersection:
    def __call__(self, arg0: sparse_vector) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def alpha(self) -> vector:
        ...
    @property
    def b(self) -> float:
        ...
    @property
    def basis_vectors(self) -> vectors:
        ...
    @property
    def kernel_function(self) -> ...:
        ...
class _decision_function_sparse_linear:
    def __call__(self, arg0: sparse_vector) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def bias(self) -> float:
        ...
    @bias.setter
    def bias(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def weights(self) -> sparse_vector:
        ...
class _decision_function_sparse_polynomial:
    def __call__(self, arg0: sparse_vector) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def alpha(self) -> vector:
        ...
    @property
    def b(self) -> float:
        ...
    @property
    def basis_vectors(self) -> vectors:
        ...
    @property
    def kernel_function(self) -> ...:
        ...
class _decision_function_sparse_radial_basis:
    def __call__(self, arg0: sparse_vector) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def alpha(self) -> vector:
        ...
    @property
    def b(self) -> float:
        ...
    @property
    def basis_vectors(self) -> vectors:
        ...
    @property
    def kernel_function(self) -> ...:
        ...
class _decision_function_sparse_sigmoid:
    def __call__(self, arg0: sparse_vector) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def alpha(self) -> vector:
        ...
    @property
    def b(self) -> float:
        ...
    @property
    def basis_vectors(self) -> vectors:
        ...
    @property
    def kernel_function(self) -> ...:
        ...
class _linear_kernel:
    def __repr__(self) -> str:
        ...
class _normalized_decision_function_radial_basis:
    @typing.overload
    def __call__(self, arg0: vector) -> float:
        ...
    @typing.overload
    def __call__(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @typing.overload
    def batch_predict(self, arg0: vectors) -> array:
        ...
    @typing.overload
    def batch_predict(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def alpha(self) -> vector:
        ...
    @property
    def b(self) -> float:
        ...
    @property
    def basis_vectors(self) -> vectors:
        ...
    @property
    def invstd_devs(self) -> vector:
        """
        Input vectors are normalized by the equation, (x-means)*invstd_devs, before being passed to the underlying RBF function.
        """
    @property
    def kernel_function(self) -> _radial_basis_kernel:
        ...
    @property
    def means(self) -> vector:
        """
        Input vectors are normalized by the equation, (x-means)*invstd_devs, before being passed to the underlying RBF function.
        """
class _radial_basis_kernel:
    def __repr__(self) -> str:
        ...
    @property
    def gamma(self) -> float:
        ...
class _range_iter:
    def __next__(self) -> int:
        ...
    def next(self) -> int:
        ...
class _ranking_test:
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def mean_ap(self) -> float:
        """
        A value between 0 and 1, measures the mean average precision of the ranking.
        """
    @mean_ap.setter
    def mean_ap(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ranking_accuracy(self) -> float:
        """
        A value between 0 and 1, measures the fraction of times a relevant sample was ordered before a non-relevant sample.
        """
    @ranking_accuracy.setter
    def ranking_accuracy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class _regression_test:
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def R_squared(self) -> float:
        """
        A value between 0 and 1, measures the squared correlation between the output of a 
        regression function and the target values.
        """
    @R_squared.setter
    def R_squared(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mean_average_error(self) -> float:
        """
        The mean average error of a regression function on a dataset.
        """
    @mean_average_error.setter
    def mean_average_error(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mean_error_stddev(self) -> float:
        """
        The standard deviation of the absolute value of the error of a regression function on a dataset.
        """
    @mean_error_stddev.setter
    def mean_error_stddev(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def mean_squared_error(self) -> float:
        """
        The mean squared error of a regression function on a dataset.
        """
    @mean_squared_error.setter
    def mean_squared_error(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class _row:
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> float:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
class array:
    """
    This object represents a 1D array of floating point numbers. Moreover, it binds directly to the C++ type std::vector<double>.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: typing.SupportsFloat | typing.SupportsIndex) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: array) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> array:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: array) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing.Any) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[float]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: array) -> bool:
        ...
    @typing.overload
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: array) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    def append(self, x: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: typing.SupportsFloat | typing.SupportsIndex) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: array) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> float:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class cca_outputs:
    Ltrans: matrix
    Rtrans: matrix
    correlations: vector
class chip_details:
    """
    WHAT THIS OBJECT REPRESENTS 
        This object describes where an image chip is to be extracted from within 
        another image.  In particular, it specifies that the image chip is 
        contained within the rectangle self.rect and that prior to extraction the 
        image should be rotated counter-clockwise by self.angle radians.  Finally, 
        the extracted chip should have self.rows rows and self.cols columns in it 
        regardless of the shape of self.rect.  This means that the extracted chip 
        will be stretched to fit via bilinear interpolation when necessary.
    """
    rect: drectangle
    @typing.overload
    def __init__(self, rect: drectangle) -> None:
        ...
    @typing.overload
    def __init__(self, rect: rectangle) -> None:
        """
        ensures 
            - self.rect == rect_ 
            - self.angle == 0 
            - self.rows == rect.height() 
            - self.cols == rect.width()
        """
    @typing.overload
    def __init__(self, rect: drectangle, size: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, rect: rectangle, size: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        ensures 
            - self.rect == rect 
            - self.angle == 0 
            - self.rows and self.cols is set such that the total size of the chip is as close 
              to size as possible but still matches the aspect ratio of rect. 
            - As long as size and the aspect ratio of rect stays constant then 
              self.rows and self.cols will always have the same values.  This means 
              that, for example, if you want all your chips to have the same dimensions 
              then ensure that size is always the same and also that rect always has 
              the same aspect ratio.  Otherwise the calculated values of self.rows and 
              self.cols may be different for different chips.  Alternatively, you can 
              use the chip_details constructor below that lets you specify the exact 
              values for rows and cols.
        """
    @typing.overload
    def __init__(self, rect: drectangle, size: typing.SupportsInt | typing.SupportsIndex, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, rect: rectangle, size: typing.SupportsInt | typing.SupportsIndex, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        ensures 
            - self.rect == rect 
            - self.angle == angle 
            - self.rows and self.cols is set such that the total size of the chip is as 
              close to size as possible but still matches the aspect ratio of rect. 
            - As long as size and the aspect ratio of rect stays constant then 
              self.rows and self.cols will always have the same values.  This means 
              that, for example, if you want all your chips to have the same dimensions 
              then ensure that size is always the same and also that rect always has 
              the same aspect ratio.  Otherwise the calculated values of self.rows and 
              self.cols may be different for different chips.  Alternatively, you can 
              use the chip_details constructor below that lets you specify the exact 
              values for rows and cols.
        """
    @typing.overload
    def __init__(self, rect: drectangle, dims: chip_dims) -> None:
        ...
    @typing.overload
    def __init__(self, rect: rectangle, dims: chip_dims) -> None:
        """
        ensures 
            - self.rect == rect 
            - self.angle == 0 
            - self.rows == dims.rows 
            - self.cols == dims.cols
        """
    @typing.overload
    def __init__(self, rect: drectangle, dims: chip_dims, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, rect: rectangle, dims: chip_dims, angle: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        ensures 
            - self.rect == rect 
            - self.angle == angle 
            - self.rows == dims.rows 
            - self.cols == dims.cols
        """
    @typing.overload
    def __init__(self, chip_points: dpoints, img_points: dpoints, dims: chip_dims) -> None:
        ...
    @typing.overload
    def __init__(self, chip_points: points, img_points: points, dims: chip_dims) -> None:
        """
        requires 
            - len(chip_points) == len(img_points) 
            - len(chip_points) >= 2  
        ensures 
            - The chip will be extracted such that the pixel locations chip_points[i] 
              in the chip are mapped to img_points[i] in the original image by a 
              similarity transform.  That is, if you know the pixelwize mapping you 
              want between the chip and the original image then you use this function 
              of chip_details constructor to define the mapping. 
            - self.rows == dims.rows 
            - self.cols == dims.cols 
            - self.rect and self.angle are computed based on the given size of the output chip 
              (specified by dims) and the similarity transform between the chip and 
              image (specified by chip_points and img_points).
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def angle(self) -> float:
        ...
    @angle.setter
    def angle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cols(self) -> int:
        ...
    @cols.setter
    def cols(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def rows(self) -> int:
        ...
    @rows.setter
    def rows(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class chip_detailss:
    """
    An array of chip_details objects.
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
    def __getitem__(self, s: slice) -> chip_detailss:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> chip_details:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: chip_detailss) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[chip_details]:
        ...
    def __len__(self) -> int:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: chip_details) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: chip_detailss) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: chip_details) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def extend(self, L: chip_detailss) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self: ..., std: ..., std: ..., std: ..., arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: chip_details) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> chip_details:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> chip_details:
        """
        Remove and return the item at index ``i``
        """
class chip_dims:
    """
    WHAT THIS OBJECT REPRESENTS 
        This is a simple tool for passing in a pair of row and column values to the 
        chip_details constructor.
    """
    def __init__(self, rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def cols(self) -> int:
        ...
    @cols.setter
    def cols(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def rows(self) -> int:
        ...
    @rows.setter
    def rows(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class cnn_face_detection_model_v1:
    """
    This object detects human faces in an image.  The constructor loads the face detection model from a file. You can download a pre-trained model from http://dlib.net/files/mmod_human_face_detector.dat.bz2.
    """
    @typing.overload
    def __call__(self, imgs: list, upsample_num_times: typing.SupportsInt | typing.SupportsIndex = 0, batch_size: typing.SupportsInt | typing.SupportsIndex = 128) -> ...:
        """
        takes a list of images as input returning a 2d list of mmod rectangles
        """
    @typing.overload
    def __call__(self, img: numpy.ndarray, upsample_num_times: typing.SupportsInt | typing.SupportsIndex = 0) -> ...:
        """
        Find faces in an image using a deep learning model.
                  - Upsamples the image upsample_num_times before running the face 
                    detector.
        """
    def __init__(self, filename: str) -> None:
        ...
class correlation_tracker:
    """
    This is a tool for tracking moving objects in a video stream.  You give it 
                the bounding box of an object in the first frame and it attempts to track the 
                object in the box from frame to frame.  
                This tool is an implementation of the method described in the following paper: 
                    Danelljan, Martin, et al. 'Accurate scale estimation for robust visual 
                    tracking.' Proceedings of the British Machine Vision Conference BMVC. 2014.
    """
    def __init__(self) -> None:
        ...
    def get_position(self) -> drectangle:
        """
        returns the predicted position of the object under track.
        """
    @typing.overload
    def start_track(self, image: numpy.ndarray, bounding_box: drectangle) -> None:
        """
                    requires 
                        - image is a numpy ndarray containing either an 8bit grayscale or RGB image. 
                        - bounding_box.is_empty() == false 
                    ensures 
                        - This object will start tracking the thing inside the bounding box in the 
                          given image.  That is, if you call update() with subsequent video frames 
                          then it will try to keep track of the position of the object inside bounding_box. 
                        - #get_position() == bounding_box
        """
    @typing.overload
    def start_track(self, image: numpy.ndarray, bounding_box: rectangle) -> None:
        """
                    requires 
                        - image is a numpy ndarray containing either an 8bit grayscale or RGB image. 
                        - bounding_box.is_empty() == false 
                    ensures 
                        - This object will start tracking the thing inside the bounding box in the 
                          given image.  That is, if you call update() with subsequent video frames 
                          then it will try to keep track of the position of the object inside bounding_box. 
                        - #get_position() == bounding_box
        """
    @typing.overload
    def update(self, image: numpy.ndarray) -> float:
        """
                    requires 
                        - image is a numpy ndarray containing either an 8bit grayscale or RGB image. 
                        - get_position().is_empty() == false 
                          (i.e. you must have started tracking by calling start_track()) 
                    ensures 
                        - performs: return update(img, get_position())
        """
    @typing.overload
    def update(self, image: numpy.ndarray, guess: drectangle) -> float:
        """
                    requires 
                        - image is a numpy ndarray containing either an 8bit grayscale or RGB image. 
                        - get_position().is_empty() == false 
                          (i.e. you must have started tracking by calling start_track()) 
                    ensures 
                        - When searching for the object in img, we search in the area around the 
                          provided guess. 
                        - #get_position() == the new predicted location of the object in img.  This 
                          location will be a copy of guess that has been translated and scaled 
                          appropriately based on the content of img so that it, hopefully, bounds 
                          the object in img. 
                        - Returns the peak to side-lobe ratio.  This is a number that measures how 
                          confident the tracker is that the object is inside #get_position(). 
                          Larger values indicate higher confidence.
        """
    @typing.overload
    def update(self, image: numpy.ndarray, guess: rectangle) -> float:
        """
                    requires 
                        - image is a numpy ndarray containing either an 8bit grayscale or RGB image. 
                        - get_position().is_empty() == false 
                          (i.e. you must have started tracking by calling start_track()) 
                    ensures 
                        - When searching for the object in img, we search in the area around the 
                          provided guess. 
                        - #get_position() == the new predicted location of the object in img.  This 
                          location will be a copy of guess that has been translated and scaled 
                          appropriately based on the content of img so that it, hopefully, bounds 
                          the object in img. 
                        - Returns the peak to side-lobe ratio.  This is a number that measures how 
                          confident the tracker is that the object is inside #get_position(). 
                          Larger values indicate higher confidence.
        """
class dpoint:
    """
    This object represents a single point of floating point coordinates that maps directly to a dlib::dpoint.
    """
    def __add__(self, arg0: dpoint) -> dpoint:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self, x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, p: point) -> None:
        ...
    @typing.overload
    def __init__(self, v: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> None:
        ...
    @typing.overload
    def __init__(self, v: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        ...
    @typing.overload
    def __init__(self, v: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        ...
    def __mul__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> dpoint:
        ...
    def __repr__(self) -> str:
        ...
    def __rmul__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> dpoint:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __sub__(self, arg0: dpoint) -> dpoint:
        ...
    def __truediv__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> dpoint:
        ...
    def normalize(self) -> dpoint:
        """
        Returns a unit normalized copy of this vector.
        """
    @property
    def x(self) -> float:
        """
        The x-coordinate of the dpoint.
        """
    @x.setter
    def x(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def y(self) -> float:
        """
        The y-coordinate of the dpoint.
        """
    @y.setter
    def y(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class dpoints:
    """
    An array of dpoint objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: dpoint) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: dpoints) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> dpoints:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> dpoint:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: dpoints) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    @typing.overload
    def __init__(self, initial_size: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[dpoint]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: dpoints) -> bool:
        ...
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dpoint) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: dpoints) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: dpoint) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: dpoint) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: dpoints) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: dpoint) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> dpoint:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> dpoint:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: dpoint) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class drectangle:
    """
    This object represents a rectangular area of an image with floating point coordinates.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: drectangle) -> bool:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self, left: typing.SupportsFloat | typing.SupportsIndex, top: typing.SupportsFloat | typing.SupportsIndex, right: typing.SupportsFloat | typing.SupportsIndex, bottom: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, rect: rectangle) -> None:
        ...
    @typing.overload
    def __init__(self, rect: drectangle) -> None:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    def __ne__(self, arg0: drectangle) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    def area(self) -> float:
        ...
    def bl_corner(self) -> dpoint:
        """
        Returns the bottom left corner of the rectangle.
        """
    def bottom(self) -> float:
        ...
    def br_corner(self) -> dpoint:
        """
        Returns the bottom right corner of the rectangle.
        """
    def center(self) -> point:
        ...
    @typing.overload
    def contains(self, point: point) -> bool:
        ...
    @typing.overload
    def contains(self, point: dpoint) -> bool:
        ...
    @typing.overload
    def contains(self, x: typing.SupportsInt | typing.SupportsIndex, y: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    @typing.overload
    def contains(self, rectangle: drectangle) -> bool:
        ...
    def dcenter(self) -> point:
        ...
    def height(self) -> float:
        ...
    def intersect(self, rectangle: drectangle) -> drectangle:
        ...
    def is_empty(self) -> bool:
        ...
    def left(self) -> float:
        ...
    def right(self) -> float:
        ...
    def tl_corner(self) -> dpoint:
        """
        Returns the top left corner of the rectangle.
        """
    def top(self) -> float:
        ...
    def tr_corner(self) -> dpoint:
        """
        Returns the top right corner of the rectangle.
        """
    def width(self) -> float:
        ...
class face_recognition_model_v1:
    """
    This object maps human faces into 128D vectors where pictures of the same person are mapped near to each other and pictures of different people are mapped far apart.  The constructor loads the face recognition model from a file. The model file is available here: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
    """
    def __init__(self, arg0: str) -> None:
        ...
    @typing.overload
    def compute_face_descriptor(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], face: full_object_detection, num_jitters: typing.SupportsInt | typing.SupportsIndex = 0, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> vector:
        """
        Takes an image and a full_object_detection that references a face in that image and converts it into a 128D face descriptor. If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. Optionally allows to override default padding of 0.25 around the face.
        """
    @typing.overload
    def compute_face_descriptor(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], num_jitters: typing.SupportsInt | typing.SupportsIndex = 0) -> vector:
        """
        Takes an aligned face image of size 150x150 and converts it into a 128D face descriptor.Note that the alignment should be done in the same way dlib.get_face_chip does it.If num_jitters>1 then image will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. 
        """
    @typing.overload
    def compute_face_descriptor(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], faces: full_object_detections, num_jitters: typing.SupportsInt | typing.SupportsIndex = 0, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> vectors:
        """
        Takes an image and an array of full_object_detections that reference faces in that image and converts them into 128D face descriptors.  If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. Optionally allows to override default padding of 0.25 around the face.
        """
    @typing.overload
    def compute_face_descriptor(self, batch_img: collections.abc.Sequence[numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]], batch_faces: collections.abc.Sequence[full_object_detections], num_jitters: typing.SupportsInt | typing.SupportsIndex = 0, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> vectorss:
        """
        Takes an array of images and an array of arrays of full_object_detections. `batch_faces[i]` must be an array of full_object_detections corresponding to the image `batch_img[i]`, referencing faces in that image. Every face will be converted into 128D face descriptors.  If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. Optionally allows to override default padding of 0.25 around the face.
        """
    @typing.overload
    def compute_face_descriptor(self, batch_img: collections.abc.Sequence[numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]], num_jitters: typing.SupportsInt | typing.SupportsIndex = 0) -> vectors:
        """
        Takes an array of aligned images of faces of size 150_x_150.Note that the alignment should be done in the same way dlib.get_face_chip does it.Every face will be converted into 128D face descriptors.  If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. 
        """
class fhog_object_detector:
    """
    This object represents a sliding window histogram-of-oriented-gradients based object detector.
    """
    @staticmethod
    def run_multiple(detectors: list, image: numpy.ndarray, upsample_num_times: typing.SupportsInt | typing.SupportsIndex = 0, adjust_threshold: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> tuple:
        """
        requires 
            - detectors is a list of detectors. 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
            - upsample_num_times >= 0 
        ensures 
            - This function runs the list of object detectors at once on the input image and returns 
              a tuple of (list of detections, list of scores, list of weight_indices).   
            - Upsamples the image upsample_num_times before running the basic 
              detector.
        """
    def __call__(self, image: numpy.ndarray, upsample_num_times: typing.SupportsInt | typing.SupportsIndex = 0) -> rectangles:
        """
        requires 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
            - upsample_num_times >= 0 
        ensures 
            - This function runs the object detector on the input image and returns 
              a list of detections.   
            - Upsamples the image upsample_num_times before running the basic 
              detector.
        """
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, arg0: str) -> None:
        """
        Loads an object detector from a file that contains the output of the 
        train_simple_object_detector() routine or a serialized C++ object of type
        object_detector<scan_fhog_pyramid<pyramid_down<6>>>.
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def run(self, image: numpy.ndarray, upsample_num_times: typing.SupportsInt | typing.SupportsIndex = 0, adjust_threshold: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> tuple:
        """
        requires 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
            - upsample_num_times >= 0 
        ensures 
            - This function runs the object detector on the input image and returns 
              a tuple of (list of detections, list of scores, list of weight_indices).   
            - Upsamples the image upsample_num_times before running the basic 
              detector.
        """
    def save(self, detector_output_filename: str) -> None:
        """
        Save a simple_object_detector to the provided path.
        """
    @property
    def detection_window_height(self) -> int:
        ...
    @property
    def detection_window_width(self) -> int:
        ...
    @property
    def num_detectors(self) -> int:
        ...
class full_object_detection:
    """
    This object represents the location of an object in an image along with the     positions of each of its constituent parts.
    """
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, rect: rectangle, parts: typing.Any) -> None:
        """
        requires 
            - rect: dlib rectangle 
            - parts: list of dlib.point, or a dlib.points object.
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def part(self, idx: typing.SupportsInt | typing.SupportsIndex) -> point:
        """
        A single part of the object as a dlib point.
        """
    def parts(self) -> points:
        """
        A vector of dlib points representing all of the parts.
        """
    @property
    def num_parts(self) -> int:
        """
        The number of parts of the object.
        """
    @property
    def rect(self) -> rectangle:
        """
        Bounding box from the underlying detector. Parts can be outside box if appropriate.
        """
class full_object_detections:
    """
    An array of full_object_detection objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: full_object_detection) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: full_object_detections) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> full_object_detections:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> full_object_detection:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: full_object_detections) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[full_object_detection]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: full_object_detections) -> bool:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: full_object_detection) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: full_object_detections) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: full_object_detection) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: full_object_detection) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: full_object_detections) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: full_object_detection) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> full_object_detection:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> full_object_detection:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: full_object_detection) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class function_evaluation:
    """
      
    This object records the output of a real valued function in response to
    some input. 
    
    In particular, if you have a function F(x) then the function_evaluation is
    simply a struct that records x and the scalar value F(x). 
    """
    @typing.overload
    def __init__(self, x: vector, y: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, x: list, y: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def x(self) -> vector:
        ...
    @property
    def y(self) -> float:
        ...
class function_evaluation_request:
    """
    See: http://dlib.net/dlib/global_optimization/global_function_search_abstract.h.html
    """
    def set(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def function_idx(self) -> int:
        ...
    @property
    def has_been_evaluated(self) -> bool:
        ...
    @property
    def x(self) -> vector:
        ...
class function_spec:
    """
    See: http://dlib.net/dlib/global_optimization/global_function_search_abstract.h.html
    """
    @typing.overload
    def __init__(self, bound1: vector, bound2: vector) -> None:
        ...
    @typing.overload
    def __init__(self, bound1: vector, bound2: vector, is_integer: collections.abc.Sequence[bool]) -> None:
        ...
    @typing.overload
    def __init__(self, bound1: list, bound2: list) -> None:
        ...
    @typing.overload
    def __init__(self, bound1: list, bound2: list, is_integer: list) -> None:
        ...
    @property
    def is_integer_variable(self) -> list[bool]:
        ...
    @property
    def lower(self) -> vector:
        ...
    @property
    def upper(self) -> vector:
        ...
class global_function_search:
    """
    See: http://dlib.net/dlib/global_optimization/global_function_search_abstract.h.html
    """
    @typing.overload
    def __init__(self, function: function_spec) -> None:
        ...
    @typing.overload
    def __init__(self, functions: list) -> None:
        ...
    @typing.overload
    def __init__(self, functions: list, initial_function_evals: list, relative_noise_magnitude: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def get_best_function_eval(self) -> tuple[vector, float, int]:
        ...
    def get_function_evaluations(self) -> tuple[list, list]:
        ...
    def get_monte_carlo_upper_bound_sample_num(self) -> int:
        ...
    def get_next_x(self) -> function_evaluation_request:
        ...
    def get_pure_random_search_probability(self) -> float:
        ...
    def get_relative_noise_magnitude(self) -> float:
        ...
    def get_solver_epsilon(self) -> float:
        ...
    def num_functions(self) -> int:
        ...
    def set_monte_carlo_upper_bound_sample_num(self, num: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_pure_random_search_probability(self, prob: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_relative_noise_magnitude(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_seed(self, seed: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_solver_epsilon(self, eps: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class hough_transform:
    """
    This object is a tool for computing the line finding version of the Hough transform 
    given some kind of edge detection image as input.  It also allows the edge pixels 
    to be weighted such that higher weighted edge pixels contribute correspondingly 
    more to the output of the Hough transform, allowing stronger edges to create 
    correspondingly stronger line detections in the final Hough transform.
    """
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], box: rectangle) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], box: rectangle) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], box: rectangle) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], box: rectangle) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], box: rectangle) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], box: rectangle) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], box: rectangle) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], box: rectangle) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], box: rectangle) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], box: rectangle) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """
        requires 
            - box.width() == size 
            - box.height() == size 
        ensures 
            - Computes the Hough transform of the part of img contained within box. 
              In particular, we do a grayscale version of the Hough transform where any 
              non-zero pixel in img is treated as a potential component of a line and 
              accumulated into the returned Hough accumulator image.  However, rather than 
              adding 1 to each relevant accumulator bin we add the value of the pixel 
              in img to each Hough accumulator bin.  This means that, if all the 
              pixels in img are 0 or 1 then this routine performs a normal Hough 
              transform.  However, if some pixels have larger values then they will be 
              weighted correspondingly more in the resulting Hough transform. 
            - The returned hough transform image will be size rows by size columns. 
            - The returned image is the Hough transform of the part of img contained in 
              box.  Each point in the Hough image corresponds to a line in the input box. 
              In particular, the line for hough_image[y][x] is given by get_line(point(x,y)).  
              Also, when viewing the Hough image, the x-axis gives the angle of the line 
              and the y-axis the distance of the line from the center of the box.  The 
              conversion between Hough coordinates and angle and pixel distance can be 
              obtained by calling get_line_properties().
        """
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """
            simply performs: return self(img, get_rect(img)).  That is, just runs the hough transform on the whole input image.
        """
    def __init__(self, size_: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        requires 
            - size_ > 0 
        ensures 
            - This object will compute Hough transforms that are size_ by size_ pixels.   
              This is in terms of both the Hough accumulator array size as well as the 
              input image size. 
            - size() == size_
        """
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], box: rectangle, hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], box: rectangle, hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], box: rectangle, hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], box: rectangle, hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], box: rectangle, hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], box: rectangle, hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], box: rectangle, hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], box: rectangle, hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], box: rectangle, hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], box: rectangle, hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        """
        requires 
            - box.width() == size 
            - box.height() == size 
            - for all valid i: 
                - rectangle(0,0,size-1,size-1).contains(hough_points[i]) == true 
                  (i.e. hough_points must contain points in the output Hough transform 
                  space generated by this object.) 
            - angle_window_size >= 1 
            - radius_window_size >= 1 
        ensures 
            - This function computes the Hough transform of the part of img contained 
              within box.  It does the same computation as __call__() defined above, 
              except instead of accumulating into an image we create an explicit list 
              of all the points in img that contributed to each line (i.e each point in 
              the Hough image). To do this we take a list of Hough points as input and 
              only record hits on these specifically identified Hough points.  A 
              typical use of find_pixels_voting_for_lines() is to first run the normal 
              Hough transform using __call__(), then find the lines you are interested 
              in, and then call find_pixels_voting_for_lines() to determine which 
              pixels in the input image belong to those lines. 
            - This routine returns a vector, CONSTITUENT_POINTS, with the following 
              properties: 
                - CONSTITUENT_POINTS.size == hough_points.size 
                - for all valid i: 
                    - Let HP[i] = centered_rect(hough_points[i], angle_window_size, radius_window_size) 
                    - Any point in img with a non-zero value that lies on a line 
                      corresponding to one of the Hough points in HP[i] is added to 
                      CONSTITUENT_POINTS[i].  Therefore, when this routine finishes, 
                      #CONSTITUENT_POINTS[i] will contain all the points in img that 
                      voted for the lines associated with the Hough accumulator bins in 
                      HP[i]. 
                    - #CONSTITUENT_POINTS[i].size == the number of points in img that 
                      voted for any of the lines HP[i] in Hough space.  Note, however, 
                      that if angle_window_size or radius_window_size are made so large 
                      that HP[i] overlaps HP[j] for i!=j then the overlapping regions 
                      of Hough space are assigned to HP[i] or HP[j] arbitrarily. 
                      That is, we treat HP[i] and HP[j] as disjoint even if their boxes 
                      overlap.  In this case, the overlapping region is assigned to 
                      either HP[i] or HP[j] in an arbitrary manner.
        """
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        ...
    @typing.overload
    def find_pixels_voting_for_lines(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], hough_points: points, angle_window_size: typing.SupportsInt | typing.SupportsIndex = 1, radius_window_size: typing.SupportsInt | typing.SupportsIndex = 1) -> list:
        """
            performs: return find_pixels_voting_for_lines(img, get_rect(img), hough_points, angle_window_size, radius_window_size); 
        That is, just runs the routine on the whole input image.
        """
    def find_strong_hough_points(self, himg: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], hough_count_thresh: typing.SupportsFloat | typing.SupportsIndex, angle_nms_thresh: typing.SupportsFloat | typing.SupportsIndex, radius_nms_thresh: typing.SupportsFloat | typing.SupportsIndex) -> points:
        """
        requires 
            - himg has size() rows and columns. 
            - angle_nms_thresh >= 0 
            - radius_nms_thresh >= 0 
        ensures 
            - This routine finds strong lines in a Hough transform and performs 
              non-maximum suppression on the detected lines.  Recall that each point in 
              Hough space is associated with a line. Therefore, this routine finds all 
              the pixels in himg (a Hough transform image) with values >= 
              hough_count_thresh and performs non-maximum suppression on the 
              identified list of pixels.  It does this by discarding lines that are 
              within angle_nms_thresh degrees of a stronger line or within 
              radius_nms_thresh distance (in terms of radius as defined by 
              get_line_properties()) to a stronger Hough point. 
            - The identified lines are returned as a list of coordinates in himg. 
            - The returned points are sorted so that points with larger Hough transform 
              values come first.
        """
    def get_best_hough_point(self, p: point, himg: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> point:
        """
        requires 
            - himg has size rows and columns. 
            - rectangle(0,0,size-1,size-1).contains(p) == true 
        ensures 
            - This function interprets himg as a Hough image and p as a point in the 
              original image space.  Given this, it finds the maximum scoring line that 
              passes though p.  That is, it checks all the Hough accumulator bins in 
              himg corresponding to lines though p and returns the location with the 
              largest score.   
            - returns a point X such that get_rect(himg).contains(X) == true
        """
    @typing.overload
    def get_line(self, p: point) -> line:
        ...
    @typing.overload
    def get_line(self, p: dpoint) -> line:
        """
        requires 
            - rectangle(0,0,size-1,size-1).contains(p) == true 
              (i.e. p must be a point inside the Hough accumulator array) 
        ensures 
            - returns the line segment in the original image space corresponding 
              to Hough transform point p.  
            - The returned points are inside rectangle(0,0,size-1,size-1).
        """
    @typing.overload
    def get_line_angle_in_degrees(self, p: point) -> float:
        ...
    @typing.overload
    def get_line_angle_in_degrees(self, p: dpoint) -> float:
        """
        requires 
            - rectangle(0,0,size-1,size-1).contains(p) == true 
              (i.e. p must be a point inside the Hough accumulator array) 
        ensures 
            - returns the angle, in degrees, of the line corresponding to the Hough 
              transform point p.
        """
    @typing.overload
    def get_line_properties(self, p: point) -> tuple:
        ...
    @typing.overload
    def get_line_properties(self, p: dpoint) -> tuple:
        """
        requires 
            - rectangle(0,0,size-1,size-1).contains(p) == true 
              (i.e. p must be a point inside the Hough accumulator array) 
        ensures 
            - Converts a point in the Hough transform space into an angle, in degrees, 
              and a radius, measured in pixels from the center of the input image. 
            - let ANGLE_IN_DEGREES == the angle of the line corresponding to the Hough 
              transform point p.  Moreover: -90 <= ANGLE_IN_DEGREES < 90. 
            - RADIUS == the distance from the center of the input image, measured in 
              pixels, and the line corresponding to the Hough transform point p. 
              Moreover: -sqrt(size*size/2) <= RADIUS <= sqrt(size*size/2) 
            - returns a tuple of (ANGLE_IN_DEGREES, RADIUS)
        """
    @property
    def size(self) -> int:
        """
        returns the size of the Hough transforms generated by this object.  In particular, this object creates Hough transform images that are size by size pixels in size.
        """
class image_gradients:
    """
    This class is a tool for computing first and second derivatives of an 
    image.  It does this by fitting a quadratic surface around each pixel and 
    then computing the gradients of that quadratic surface.  For the details 
    see the paper: 
        Quadratic models for curved line detection in SAR CCD by Davis E. King 
        and Rhonda D. Phillips 
     
    This technique gives very accurate gradient estimates and is also very fast 
    since the entire gradient estimation procedure, for each type of gradient, 
    is accomplished by cross-correlating the image with a single separable 
    filter.  This means you can compute gradients at very large scales (e.g. by 
    fitting the quadratic to a large window, like a 99x99 window) and it still 
    runs very quickly.
    """
    @typing.overload
    def __init__(self, scale: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Creates this class with the provided scale. i.e. get_scale()==scale. 
        scale must be >= 1.
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Creates this class with a scale of 1. i.e. get_scale()==1
        """
    def get_scale(self) -> int:
        """
        When we estimate a gradient we do so by fitting a quadratic filter to a window of size 
        get_scale()*2+1 centered on each pixel.  Therefore, the scale parameter controls the size 
        of gradients we will find.  For example, a very large scale will cause the gradient_xx() 
        to be insensitive to high frequency noise in the image while smaller scales would be more 
        sensitive to such fluctuations in the image.
        """
    def get_x_filter(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """
        - Returns the filter used by the indicated derivative to compute the image gradient. 
          That is, the output gradients are found by cross correlating the returned filter with 
          the input image. 
        - The returned filter has get_scale()*2+1 rows and columns.
        """
    def get_xx_filter(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """
        - Returns the filter used by the indicated derivative to compute the image gradient. 
          That is, the output gradients are found by cross correlating the returned filter with 
          the input image. 
        - The returned filter has get_scale()*2+1 rows and columns.
        """
    def get_xy_filter(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """
        - Returns the filter used by the indicated derivative to compute the image gradient. 
          That is, the output gradients are found by cross correlating the returned filter with 
          the input image. 
        - The returned filter has get_scale()*2+1 rows and columns.
        """
    def get_y_filter(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """
        - Returns the filter used by the indicated derivative to compute the image gradient. 
          That is, the output gradients are found by cross correlating the returned filter with 
          the input image. 
        - The returned filter has get_scale()*2+1 rows and columns.
        """
    def get_yy_filter(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """
        - Returns the filter used by the indicated derivative to compute the image gradient. 
          That is, the output gradients are found by cross correlating the returned filter with 
          the input image. 
        - The returned filter has get_scale()*2+1 rows and columns.
        """
    @typing.overload
    def gradient_x(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], rectangle]:
        ...
    @typing.overload
    def gradient_x(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], rectangle]:
        """
        - Let VALID_AREA = shrink_rect(get_rect(img),get_scale()). 
        - This routine computes the requested gradient of img at each location in VALID_AREA. 
          The gradients are returned in a new image of the same dimensions as img.  All pixels 
          outside VALID_AREA are set to 0.  VALID_AREA is also returned.  I.e. we return a tuple 
          where the first element is the gradient image and the second is VALID_AREA.
        """
    @typing.overload
    def gradient_xx(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], rectangle]:
        ...
    @typing.overload
    def gradient_xx(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], rectangle]:
        """
        - Let VALID_AREA = shrink_rect(get_rect(img),get_scale()). 
        - This routine computes the requested gradient of img at each location in VALID_AREA. 
          The gradients are returned in a new image of the same dimensions as img.  All pixels 
          outside VALID_AREA are set to 0.  VALID_AREA is also returned.  I.e. we return a tuple 
          where the first element is the gradient image and the second is VALID_AREA.
        """
    @typing.overload
    def gradient_xy(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], rectangle]:
        ...
    @typing.overload
    def gradient_xy(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], rectangle]:
        """
        - Let VALID_AREA = shrink_rect(get_rect(img),get_scale()). 
        - This routine computes the requested gradient of img at each location in VALID_AREA. 
          The gradients are returned in a new image of the same dimensions as img.  All pixels 
          outside VALID_AREA are set to 0.  VALID_AREA is also returned.  I.e. we return a tuple 
          where the first element is the gradient image and the second is VALID_AREA.
        """
    @typing.overload
    def gradient_y(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], rectangle]:
        ...
    @typing.overload
    def gradient_y(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], rectangle]:
        """
        - Let VALID_AREA = shrink_rect(get_rect(img),get_scale()). 
        - This routine computes the requested gradient of img at each location in VALID_AREA. 
          The gradients are returned in a new image of the same dimensions as img.  All pixels 
          outside VALID_AREA are set to 0.  VALID_AREA is also returned.  I.e. we return a tuple 
          where the first element is the gradient image and the second is VALID_AREA.
        """
    @typing.overload
    def gradient_yy(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], rectangle]:
        ...
    @typing.overload
    def gradient_yy(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> tuple[numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], rectangle]:
        """
        - Let VALID_AREA = shrink_rect(get_rect(img),get_scale()). 
        - This routine computes the requested gradient of img at each location in VALID_AREA. 
          The gradients are returned in a new image of the same dimensions as img.  All pixels 
          outside VALID_AREA are set to 0.  VALID_AREA is also returned.  I.e. we return a tuple 
          where the first element is the gradient image and the second is VALID_AREA.
        """
class image_window:
    """
    This is a GUI window capable of showing images on the screen.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: fhog_object_detector) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: simple_object_detector) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: fhog_object_detector, arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: simple_object_detector, arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> None:
        """
        Create an image window that displays the given numpy image.
        """
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], arg1: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], arg1: str) -> None:
        """
        Create an image window that displays the given numpy image and also has the given title.
        """
    @typing.overload
    def add_overlay(self, rectangles: rectangles, color: rgb_pixel = ...) -> None:
        """
        Add a list of rectangles to the image_window. They will be displayed as red boxes by default, but the color can be passed.
        """
    @typing.overload
    def add_overlay(self, rectangle: rectangle, color: rgb_pixel = ...) -> None:
        """
        Add a rectangle to the image_window.  It will be displayed as a red box by default, but the color can be passed.
        """
    @typing.overload
    def add_overlay(self, rectangle: drectangle, color: rgb_pixel = ...) -> None:
        """
        Add a rectangle to the image_window.  It will be displayed as a red box by default, but the color can be passed.
        """
    @typing.overload
    def add_overlay(self, detection: full_object_detection, color: rgb_pixel = ...) -> None:
        """
        Add full_object_detection parts to the image window. They will be displayed as blue lines by default, but the color can be passed.
        """
    @typing.overload
    def add_overlay(self, line: line, color: rgb_pixel = ...) -> None:
        """
        Add line to the image window.
        """
    @typing.overload
    def add_overlay(self, objects: list, color: rgb_pixel = ...) -> None:
        """
        Adds all the overlayable objects, uses the given color.
        """
    @typing.overload
    def add_overlay_circle(self, center: point, radius: typing.SupportsFloat | typing.SupportsIndex, color: rgb_pixel = ...) -> None:
        """
        Add circle to the image window.
        """
    @typing.overload
    def add_overlay_circle(self, center: dpoint, radius: typing.SupportsFloat | typing.SupportsIndex, color: rgb_pixel = ...) -> None:
        """
        Add circle to the image window.
        """
    def clear_overlay(self) -> None:
        """
        Remove all overlays from the image_window.
        """
    def get_next_double_click(self) -> typing.Any:
        """
        Blocks until the user double clicks on the image or closes the window.  Returns a dlib.point indicating the pixel the user clicked on or None if the window as closed.
        """
    def get_next_keypress(self, get_keyboard_modifiers: bool = False) -> typing.Any:
        """
        Blocks until the user presses a key on their keyboard or the window is closed. 
         
        ensures 
            - if (get_keyboard_modifiers==True) then 
                - returns a tuple of (key_pressed, keyboard_modifiers_active) 
            - else 
                - returns just the key that was pressed.   
            - The returned key is either a str containing the letter that was pressed, or  
              an element of the dlib.non_printable_keyboard_keys enum. 
            - keyboard_modifiers_active, if returned, is a list of elements of the 
              dlib.keyboard_mod_keys enum.  They tell you if a key like shift was being held 
              down or not during the button press. 
            - If the window is closed before the user presses a key then this function 
              returns with all outputs set to None.
        """
    def is_closed(self) -> bool:
        """
        returns true if this window has been closed, false otherwise.  (Note that closed windows do not receive any callbacks at all.  They are also not visible on the screen.)
        """
    @typing.overload
    def set_image(self, detector: simple_object_detector) -> None:
        """
        Make the image_window display the given HOG detector's filters.
        """
    @typing.overload
    def set_image(self, detector: fhog_object_detector) -> None:
        """
        Make the image_window display the given HOG detector's filters.
        """
    @typing.overload
    def set_image(self, image: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> None:
        ...
    @typing.overload
    def set_image(self, image: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> None:
        ...
    @typing.overload
    def set_image(self, image: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> None:
        ...
    @typing.overload
    def set_image(self, image: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]) -> None:
        ...
    @typing.overload
    def set_image(self, image: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]) -> None:
        ...
    @typing.overload
    def set_image(self, image: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]) -> None:
        ...
    @typing.overload
    def set_image(self, image: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]) -> None:
        ...
    @typing.overload
    def set_image(self, image: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]) -> None:
        ...
    @typing.overload
    def set_image(self, image: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> None:
        ...
    @typing.overload
    def set_image(self, image: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
        ...
    @typing.overload
    def set_image(self, image: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> None:
        """
        Make the image_window display the given image.
        """
    def set_title(self, title: str) -> None:
        """
        Set the title of the window to the given value.
        """
    @typing.overload
    def wait_for_keypress(self, key: str) -> None:
        """
        Blocks until the user presses the given key or closes the window.
        """
    @typing.overload
    def wait_for_keypress(self, key: ...) -> None:
        """
        Blocks until the user presses the given key or closes the window.
        """
    def wait_until_closed(self) -> None:
        """
        This function blocks until the window is closed.
        """
class keyboard_mod_keys:
    """
    Members:
    
      KBD_MOD_NONE
    
      KBD_MOD_SHIFT
    
      KBD_MOD_CONTROL
    
      KBD_MOD_ALT
    
      KBD_MOD_META
    
      KBD_MOD_CAPS_LOCK
    
      KBD_MOD_NUM_LOCK
    
      KBD_MOD_SCROLL_LOCK
    """
    KBD_MOD_ALT: typing.ClassVar[keyboard_mod_keys]  # value = <keyboard_mod_keys.KBD_MOD_ALT: 4>
    KBD_MOD_CAPS_LOCK: typing.ClassVar[keyboard_mod_keys]  # value = <keyboard_mod_keys.KBD_MOD_CAPS_LOCK: 16>
    KBD_MOD_CONTROL: typing.ClassVar[keyboard_mod_keys]  # value = <keyboard_mod_keys.KBD_MOD_CONTROL: 2>
    KBD_MOD_META: typing.ClassVar[keyboard_mod_keys]  # value = <keyboard_mod_keys.KBD_MOD_META: 8>
    KBD_MOD_NONE: typing.ClassVar[keyboard_mod_keys]  # value = <keyboard_mod_keys.KBD_MOD_NONE: 0>
    KBD_MOD_NUM_LOCK: typing.ClassVar[keyboard_mod_keys]  # value = <keyboard_mod_keys.KBD_MOD_NUM_LOCK: 32>
    KBD_MOD_SCROLL_LOCK: typing.ClassVar[keyboard_mod_keys]  # value = <keyboard_mod_keys.KBD_MOD_SCROLL_LOCK: 64>
    KBD_MOD_SHIFT: typing.ClassVar[keyboard_mod_keys]  # value = <keyboard_mod_keys.KBD_MOD_SHIFT: 1>
    __members__: typing.ClassVar[dict[str, keyboard_mod_keys]]  # value = {'KBD_MOD_NONE': <keyboard_mod_keys.KBD_MOD_NONE: 0>, 'KBD_MOD_SHIFT': <keyboard_mod_keys.KBD_MOD_SHIFT: 1>, 'KBD_MOD_CONTROL': <keyboard_mod_keys.KBD_MOD_CONTROL: 2>, 'KBD_MOD_ALT': <keyboard_mod_keys.KBD_MOD_ALT: 4>, 'KBD_MOD_META': <keyboard_mod_keys.KBD_MOD_META: 8>, 'KBD_MOD_CAPS_LOCK': <keyboard_mod_keys.KBD_MOD_CAPS_LOCK: 16>, 'KBD_MOD_NUM_LOCK': <keyboard_mod_keys.KBD_MOD_NUM_LOCK: 32>, 'KBD_MOD_SCROLL_LOCK': <keyboard_mod_keys.KBD_MOD_SCROLL_LOCK: 64>}
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
class line:
    """
    This object represents a line in the 2D plane.  The line is defined by two points 
    running through it, p1 and p2.  This object also includes a unit normal vector that 
    is perpendicular to the line.
    """
    @typing.overload
    def __init__(self) -> None:
        """
        p1, p2, and normal are all the 0 vector.
        """
    @typing.overload
    def __init__(self, a: dpoint, b: dpoint) -> None:
        """
        ensures 
            - #p1 == a 
            - #p2 == b 
            - #normal == A vector normal to the line passing through points a and b. 
              Therefore, the normal vector is the vector (a-b) but unit normalized and rotated clockwise 90 degrees.
        """
    @typing.overload
    def __init__(self, a: point, b: point) -> None:
        """
        ensures 
            - #p1 == a 
            - #p2 == b 
            - #normal == A vector normal to the line passing through points a and b. 
              Therefore, the normal vector is the vector (a-b) but unit normalized and rotated clockwise 90 degrees.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def normal(self) -> dpoint:
        """
        returns a unit vector that is normal to the line passing through p1 and p2.
        """
    @property
    def p1(self) -> dpoint:
        """
        returns the first endpoint of the line.
        """
    @property
    def p2(self) -> dpoint:
        """
        returns the second endpoint of the line.
        """
class matrix:
    """
    This object represents a dense 2D matrix of floating point numbers.Moreover, it binds directly to the C++ type dlib::matrix<double>.
    """
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> _row:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: list) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing.Any) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    def deserialize(self, file: str) -> None:
        """
        Deserialize the matrix from a file
        """
    def nc(self) -> int:
        """
        Return the number of columns in the matrix.
        """
    def nr(self) -> int:
        """
        Return the number of rows in the matrix.
        """
    def serialize(self, file: str) -> None:
        """
        Serialize the matrix to a file
        """
    def set_size(self, rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the size of the matrix to the given number of rows and columns.
        """
    @property
    def shape(self) -> tuple:
        ...
class mmod_rectangle:
    """
    Wrapper around a rectangle object and a detection confidence score.
    """
    rect: rectangle
    @property
    def confidence(self) -> float:
        ...
    @confidence.setter
    def confidence(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class mmod_rectangles:
    """
    An array of mmod rectangle objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: mmod_rectangle) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: mmod_rectangles) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> mmod_rectangles:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> mmod_rectangle:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mmod_rectangles) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[mmod_rectangle]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: mmod_rectangles) -> bool:
        ...
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: mmod_rectangle) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: mmod_rectangles) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: mmod_rectangle) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: mmod_rectangle) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: mmod_rectangles) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: mmod_rectangle) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> mmod_rectangle:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> mmod_rectangle:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: mmod_rectangle) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class mmod_rectangless:
    """
    A 2D array of mmod rectangle objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: mmod_rectangles) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: mmod_rectangless) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> mmod_rectangless:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> mmod_rectangles:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mmod_rectangless) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[mmod_rectangles]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: mmod_rectangless) -> bool:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: mmod_rectangles) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: mmod_rectangless) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: mmod_rectangles) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: mmod_rectangles) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: mmod_rectangless) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: mmod_rectangles) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> mmod_rectangles:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> mmod_rectangles:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: mmod_rectangles) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class momentum_filter:
    """
     
                    This object is a simple tool for filtering a single scalar value that
                    measures the location of a moving object that has some non-trivial
                    momentum.  Importantly, the measurements are noisy and the object can
                    experience sudden unpredictable accelerations.  To accomplish this
                    filtering we use a simple Kalman filter with a state transition model of:
    
                        position_{i+1} = position_{i} + velocity_{i} 
                        velocity_{i+1} = velocity_{i} + some_unpredictable_acceleration
    
                    and a measurement model of:
                        
                        measured_position_{i} = position_{i} + measurement_noise
    
                    Where some_unpredictable_acceleration and measurement_noise are 0 mean Gaussian 
                    noise sources with standard deviations of get_typical_acceleration() and
                    get_measurement_noise() respectively.
    
                    To allow for really sudden and large but infrequent accelerations, at each
                    step we check if the current measured position deviates from the predicted
                    filtered position by more than get_max_measurement_deviation()*get_measurement_noise() 
                    and if so we adjust the filter's state to keep it within these bounds.
                    This allows the moving object to undergo large unmodeled accelerations, far
                    in excess of what would be suggested by get_typical_acceleration(), without
                    then experiencing a long lag time where the Kalman filter has to "catch
                    up" to the new position.  
    """
    def __call__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, measurement_noise: typing.SupportsFloat | typing.SupportsIndex, typical_acceleration: typing.SupportsFloat | typing.SupportsIndex, max_measurement_deviation: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def max_measurement_deviation(self) -> float:
        ...
    def measurement_noise(self) -> float:
        ...
    def typical_acceleration(self) -> float:
        ...
class no_convex_quadrilateral(Exception):
    pass
class non_printable_keyboard_keys:
    """
    Members:
    
      KEY_BACKSPACE
    
      KEY_SHIFT
    
      KEY_CTRL
    
      KEY_ALT
    
      KEY_PAUSE
    
      KEY_CAPS_LOCK
    
      KEY_ESC
    
      KEY_PAGE_UP
    
      KEY_PAGE_DOWN
    
      KEY_END
    
      KEY_HOME
    
      KEY_LEFT
    
      KEY_RIGHT
    
      KEY_UP
    
      KEY_DOWN
    
      KEY_INSERT
    
      KEY_DELETE
    
      KEY_SCROLL_LOCK
    
      KEY_F1
    
      KEY_F2
    
      KEY_F3
    
      KEY_F4
    
      KEY_F5
    
      KEY_F6
    
      KEY_F7
    
      KEY_F8
    
      KEY_F9
    
      KEY_F10
    
      KEY_F11
    
      KEY_F12
    """
    KEY_ALT: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_ALT: 3>
    KEY_BACKSPACE: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_BACKSPACE: 0>
    KEY_CAPS_LOCK: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_CAPS_LOCK: 5>
    KEY_CTRL: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_CTRL: 2>
    KEY_DELETE: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_DELETE: 16>
    KEY_DOWN: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_DOWN: 14>
    KEY_END: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_END: 9>
    KEY_ESC: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_ESC: 6>
    KEY_F1: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_F1: 18>
    KEY_F10: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_F10: 27>
    KEY_F11: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_F11: 28>
    KEY_F12: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_F12: 29>
    KEY_F2: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_F2: 19>
    KEY_F3: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_F3: 20>
    KEY_F4: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_F4: 21>
    KEY_F5: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_F5: 22>
    KEY_F6: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_F6: 23>
    KEY_F7: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_F7: 24>
    KEY_F8: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_F8: 25>
    KEY_F9: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_F9: 26>
    KEY_HOME: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_HOME: 10>
    KEY_INSERT: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_INSERT: 15>
    KEY_LEFT: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_LEFT: 11>
    KEY_PAGE_DOWN: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_PAGE_DOWN: 8>
    KEY_PAGE_UP: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_PAGE_UP: 7>
    KEY_PAUSE: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_PAUSE: 4>
    KEY_RIGHT: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_RIGHT: 12>
    KEY_SCROLL_LOCK: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_SCROLL_LOCK: 17>
    KEY_SHIFT: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_SHIFT: 1>
    KEY_UP: typing.ClassVar[non_printable_keyboard_keys]  # value = <non_printable_keyboard_keys.KEY_UP: 13>
    __members__: typing.ClassVar[dict[str, non_printable_keyboard_keys]]  # value = {'KEY_BACKSPACE': <non_printable_keyboard_keys.KEY_BACKSPACE: 0>, 'KEY_SHIFT': <non_printable_keyboard_keys.KEY_SHIFT: 1>, 'KEY_CTRL': <non_printable_keyboard_keys.KEY_CTRL: 2>, 'KEY_ALT': <non_printable_keyboard_keys.KEY_ALT: 3>, 'KEY_PAUSE': <non_printable_keyboard_keys.KEY_PAUSE: 4>, 'KEY_CAPS_LOCK': <non_printable_keyboard_keys.KEY_CAPS_LOCK: 5>, 'KEY_ESC': <non_printable_keyboard_keys.KEY_ESC: 6>, 'KEY_PAGE_UP': <non_printable_keyboard_keys.KEY_PAGE_UP: 7>, 'KEY_PAGE_DOWN': <non_printable_keyboard_keys.KEY_PAGE_DOWN: 8>, 'KEY_END': <non_printable_keyboard_keys.KEY_END: 9>, 'KEY_HOME': <non_printable_keyboard_keys.KEY_HOME: 10>, 'KEY_LEFT': <non_printable_keyboard_keys.KEY_LEFT: 11>, 'KEY_RIGHT': <non_printable_keyboard_keys.KEY_RIGHT: 12>, 'KEY_UP': <non_printable_keyboard_keys.KEY_UP: 13>, 'KEY_DOWN': <non_printable_keyboard_keys.KEY_DOWN: 14>, 'KEY_INSERT': <non_printable_keyboard_keys.KEY_INSERT: 15>, 'KEY_DELETE': <non_printable_keyboard_keys.KEY_DELETE: 16>, 'KEY_SCROLL_LOCK': <non_printable_keyboard_keys.KEY_SCROLL_LOCK: 17>, 'KEY_F1': <non_printable_keyboard_keys.KEY_F1: 18>, 'KEY_F2': <non_printable_keyboard_keys.KEY_F2: 19>, 'KEY_F3': <non_printable_keyboard_keys.KEY_F3: 20>, 'KEY_F4': <non_printable_keyboard_keys.KEY_F4: 21>, 'KEY_F5': <non_printable_keyboard_keys.KEY_F5: 22>, 'KEY_F6': <non_printable_keyboard_keys.KEY_F6: 23>, 'KEY_F7': <non_printable_keyboard_keys.KEY_F7: 24>, 'KEY_F8': <non_printable_keyboard_keys.KEY_F8: 25>, 'KEY_F9': <non_printable_keyboard_keys.KEY_F9: 26>, 'KEY_F10': <non_printable_keyboard_keys.KEY_F10: 27>, 'KEY_F11': <non_printable_keyboard_keys.KEY_F11: 28>, 'KEY_F12': <non_printable_keyboard_keys.KEY_F12: 29>}
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    @typing.overload
    def __eq__(self, arg0: str) -> bool:
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
class pair:
    """
    This object is used to represent the elements of a sparse_vector.
    """
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def first(self) -> int:
        """
        This field represents the index/dimension number.
        """
    @first.setter
    def first(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def second(self) -> float:
        """
        This field contains the value in a vector at dimension specified by the first field.
        """
    @second.setter
    def second(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class point:
    """
    This object represents a single point of integer coordinates that maps directly to a dlib::point.
    """
    @staticmethod
    @typing.overload
    def __init__(*args, **kwargs) -> None:
        ...
    def __add__(self, arg0: point) -> point:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self, x: typing.SupportsInt | typing.SupportsIndex, y: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, v: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> None:
        ...
    @typing.overload
    def __init__(self, v: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        ...
    @typing.overload
    def __init__(self, v: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        ...
    def __mul__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> ...:
        ...
    def __repr__(self) -> str:
        ...
    def __rmul__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> ...:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __sub__(self, arg0: point) -> point:
        ...
    def __truediv__(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> ...:
        ...
    def normalize(self) -> ...:
        """
        Returns a unit normalized copy of this vector.
        """
    @property
    def x(self) -> int:
        """
        The x-coordinate of the point.
        """
    @x.setter
    def x(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def y(self) -> int:
        """
        The y-coordinate of the point.
        """
    @y.setter
    def y(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class point_transform_projective:
    """
    This is an object that takes 2D points and applies a projective transformation to them.
    """
    def __call__(self, p: dpoint) -> dpoint:
        """
        ensures 
            - Applies the projective transformation defined by this object's constructor 
              to p and returns the result.  To define this precisely: 
                - let p_h == the point p in homogeneous coordinates.  That is: 
                    - p_h.x == p.x 
                    - p_h.y == p.y 
                    - p_h.z == 1  
                - let x == m*p_h  
                - Then this function returns the value x/x.z
        """
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        """
        ensures 
            - This object will perform the identity transform.  That is, given a point 
              as input it will return the same point as output.  Therefore, self.m == a 3x3 identity matrix.
        """
    @typing.overload
    def __init__(self, m: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
        """
        ensures 
            - self.m == m
        """
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def m(self) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        """
        m is the 3x3 matrix that defines the projective transformation.
        """
class points:
    """
    An array of point objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: point) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: points) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> points:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> point:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: points) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    @typing.overload
    def __init__(self, initial_size: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[point]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: points) -> bool:
        ...
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: point) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: points) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: point) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: point) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: points) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: point) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> point:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> point:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: point) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class pyramid_down:
    """
    This is a simple object to help create image pyramids.  In particular, it 
    downsamples images at a ratio of N to N-1. 
     
    Note that setting N to 1 means that this object functions like 
    pyramid_disable (defined at the bottom of this file).   
     
    WARNING, when mapping rectangles from one layer of a pyramid 
    to another you might end up with rectangles which extend slightly  
    outside your images.  This is because points on the border of an  
    image at a higher pyramid layer might correspond to points outside  
    images at lower layers.  So just keep this in mind.  Note also 
    that it's easy to deal with.  Just say something like this: 
        rect = rect.intersect(get_rect(my_image)); # keep rect inside my_image 
    """
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
        ...
    @typing.overload
    def __call__(self, img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
        """
        - Downsamples img to make a new image that is roughly (pyramid_downsampling_rate()-1)/pyramid_downsampling_rate()  
          times the size of the original image.   
        - The location of a point P in original image will show up at point point_down(P) 
          in the downsampled image.   
        - Note that some points on the border of the original image might correspond to  
          points outside the downsampled image.
        """
    @typing.overload
    def __init__(self, N: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Creates this class with the provided downsampling rate. i.e. pyramid_downsampling_rate()==N. 
        N must be in the range 1 to 20.
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Creates this class with pyramid_downsampling_rate()==2
        """
    @typing.overload
    def point_down(self, p: point) -> dpoint:
        ...
    @typing.overload
    def point_down(self, p: dpoint) -> dpoint:
        """
        Maps from pixels in a source image to the corresponding pixels in the downsampled image.
        """
    @typing.overload
    def point_down(self, p: point, levels: typing.SupportsInt | typing.SupportsIndex) -> dpoint:
        ...
    @typing.overload
    def point_down(self, p: dpoint, levels: typing.SupportsInt | typing.SupportsIndex) -> dpoint:
        """
        Applies point_down() to p levels times and returns the result.
        """
    @typing.overload
    def point_up(self, p: point) -> dpoint:
        ...
    @typing.overload
    def point_up(self, p: dpoint) -> dpoint:
        """
        Maps from pixels in a downsampled image to pixels in the original image.
        """
    @typing.overload
    def point_up(self, p: point, levels: typing.SupportsInt | typing.SupportsIndex) -> dpoint:
        ...
    @typing.overload
    def point_up(self, p: dpoint, levels: typing.SupportsInt | typing.SupportsIndex) -> dpoint:
        """
        Applies point_up() to p levels times and returns the result.
        """
    def pyramid_downsampling_rate(self) -> int:
        """
        Returns a number N that defines the downsampling rate.  In particular, images are downsampled by a factor of N to N-1.
        """
    @typing.overload
    def rect_down(self, rect: rectangle) -> rectangle:
        ...
    @typing.overload
    def rect_down(self, rect: drectangle) -> drectangle:
        """
        returns drectangle(point_down(rect.tl_corner()), point_down(rect.br_corner()));
         (i.e. maps rect into a downsampled)
        """
    @typing.overload
    def rect_down(self, rect: rectangle, levels: typing.SupportsInt | typing.SupportsIndex) -> rectangle:
        ...
    @typing.overload
    def rect_down(self, rect: drectangle, levels: typing.SupportsInt | typing.SupportsIndex) -> drectangle:
        """
        Applies rect_down() to rect levels times and returns the result.
        """
    @typing.overload
    def rect_up(self, rect: rectangle) -> rectangle:
        ...
    @typing.overload
    def rect_up(self, rect: drectangle) -> drectangle:
        """
        returns drectangle(point_up(rect.tl_corner()), point_up(rect.br_corner()));
         (i.e. maps rect into a parent image)
        """
    @typing.overload
    def rect_up(self, rect: rectangle, levels: typing.SupportsInt | typing.SupportsIndex) -> rectangle:
        ...
    @typing.overload
    def rect_up(self, p: drectangle, levels: typing.SupportsInt | typing.SupportsIndex) -> drectangle:
        """
        Applies rect_up() to rect levels times and returns the result.
        """
class range:
    """
    This object is used to represent a range of elements in an array.
    """
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __iter__(self) -> range_iter:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def begin(self) -> int:
        """
        The index of the first element in the range.  This is represented using an unsigned integer.
        """
    @begin.setter
    def begin(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def end(self) -> int:
        """
        One past the index of the last element in the range.  This is represented using an unsigned integer.
        """
    @end.setter
    def end(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class ranges:
    """
    This object is an array of range objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: range) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: ranges) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> ranges:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> range:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: ranges) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[range]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: ranges) -> bool:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: range) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: ranges) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: range) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: range) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: ranges) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: range) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> range:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> range:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: range) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class rangess:
    """
    This object is an array of arrays of range objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: ranges) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: rangess) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> rangess:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> ranges:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: rangess) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[ranges]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: rangess) -> bool:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: ranges) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: rangess) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: ranges) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: ranges) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: rangess) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: ranges) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> ranges:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> ranges:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: ranges) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class ranking_pair:
    nonrelevant: vectors
    relevant: vectors
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
class ranking_pairs:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: ranking_pair) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: ranking_pairs) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> ranking_pairs:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> ranking_pair:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: ranking_pairs) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[ranking_pair]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: ranking_pairs) -> bool:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: ranking_pair) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: ranking_pairs) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: ranking_pair) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: ranking_pair) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: ranking_pairs) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: ranking_pair) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> ranking_pair:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> ranking_pair:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: ranking_pair) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class rect_filter:
    """
     
                    This object is a simple tool for filtering a rectangle that
                    measures the location of a moving object that has some non-trivial
                    momentum.  Importantly, the measurements are noisy and the object can
                    experience sudden unpredictable accelerations.  To accomplish this
                    filtering we use a simple Kalman filter with a state transition model of:
    
                        position_{i+1} = position_{i} + velocity_{i} 
                        velocity_{i+1} = velocity_{i} + some_unpredictable_acceleration
    
                    and a measurement model of:
                        
                        measured_position_{i} = position_{i} + measurement_noise
    
                    Where some_unpredictable_acceleration and measurement_noise are 0 mean Gaussian 
                    noise sources with standard deviations of typical_acceleration and
                    measurement_noise respectively.
    
                    To allow for really sudden and large but infrequent accelerations, at each
                    step we check if the current measured position deviates from the predicted
                    filtered position by more than max_measurement_deviation*measurement_noise 
                    and if so we adjust the filter's state to keep it within these bounds.
                    This allows the moving object to undergo large unmodeled accelerations, far
                    in excess of what would be suggested by typical_acceleration, without
                    then experiencing a long lag time where the Kalman filter has to "catches
                    up" to the new position.  
    """
    def __call__(self, rect: rectangle) -> rectangle:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, measurement_noise: typing.SupportsFloat | typing.SupportsIndex, typical_acceleration: typing.SupportsFloat | typing.SupportsIndex, max_measurement_deviation: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def max_measurement_deviation(self) -> float:
        ...
    def measurement_noise(self) -> float:
        ...
    def typical_acceleration(self) -> float:
        ...
class rectangle:
    """
    This object represents a rectangular area of an image.
    """
    __hash__: typing.ClassVar[None] = None
    @typing.overload
    def __add__(self, arg0: point) -> rectangle:
        ...
    @typing.overload
    def __add__(self, arg0: rectangle) -> rectangle:
        ...
    def __eq__(self, arg0: rectangle) -> bool:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __iadd__(self, arg0: point) -> rectangle:
        ...
    @typing.overload
    def __iadd__(self, arg0: rectangle) -> rectangle:
        ...
    @typing.overload
    def __init__(self, left: typing.SupportsInt | typing.SupportsIndex, top: typing.SupportsInt | typing.SupportsIndex, right: typing.SupportsInt | typing.SupportsIndex, bottom: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, rect: ...) -> None:
        ...
    @typing.overload
    def __init__(self, rect: rectangle) -> None:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    def __ne__(self, arg0: rectangle) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    def area(self) -> int:
        ...
    def bl_corner(self) -> point:
        """
        Returns the bottom left corner of the rectangle.
        """
    def bottom(self) -> int:
        ...
    def br_corner(self) -> point:
        """
        Returns the bottom right corner of the rectangle.
        """
    def center(self) -> point:
        ...
    @typing.overload
    def contains(self, point: point) -> bool:
        ...
    @typing.overload
    def contains(self, point: dpoint) -> bool:
        ...
    @typing.overload
    def contains(self, x: typing.SupportsInt | typing.SupportsIndex, y: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    @typing.overload
    def contains(self, rectangle: rectangle) -> bool:
        ...
    def dcenter(self) -> point:
        ...
    def height(self) -> int:
        ...
    def intersect(self, rectangle: rectangle) -> rectangle:
        ...
    def is_empty(self) -> bool:
        ...
    def left(self) -> int:
        ...
    def right(self) -> int:
        ...
    def tl_corner(self) -> point:
        """
        Returns the top left corner of the rectangle.
        """
    def top(self) -> int:
        ...
    def tr_corner(self) -> point:
        """
        Returns the top right corner of the rectangle.
        """
    def width(self) -> int:
        ...
class rectangles:
    """
    An array of rectangle objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: rectangle) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: rectangles) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> rectangles:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> rectangle:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: rectangles) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    @typing.overload
    def __init__(self, initial_size: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[rectangle]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: rectangles) -> bool:
        ...
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: rectangle) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: rectangles) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: rectangle) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: rectangle) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: rectangles) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: rectangle) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> rectangle:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> rectangle:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: rectangle) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class rectangless:
    """
    An array of arrays of rectangle objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: rectangles) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: rectangless) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> rectangless:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> rectangles:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: rectangless) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    @typing.overload
    def __init__(self, initial_size: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[rectangles]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: rectangless) -> bool:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: rectangles) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: rectangless) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: rectangles) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: rectangles) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: rectangless) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self: rectangles, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: rectangles) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> rectangles:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> rectangles:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: rectangles) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class rgb_pixel:
    def __init__(self, red: typing.SupportsInt | typing.SupportsIndex, green: typing.SupportsInt | typing.SupportsIndex, blue: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def blue(self) -> int:
        ...
    @blue.setter
    def blue(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def green(self) -> int:
        ...
    @green.setter
    def green(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def red(self) -> int:
        ...
    @red.setter
    def red(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class rvm_trainer_histogram_intersection:
    def __init__(self) -> None:
        ...
    def train(self, arg0: vectors, arg1: array) -> _decision_function_histogram_intersection:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class rvm_trainer_linear:
    def __init__(self) -> None:
        ...
    def train(self, arg0: vectors, arg1: array) -> _decision_function_linear:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class rvm_trainer_radial_basis:
    def __init__(self) -> None:
        ...
    def train(self, arg0: vectors, arg1: array) -> _decision_function_radial_basis:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gamma(self) -> float:
        ...
    @gamma.setter
    def gamma(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class rvm_trainer_sparse_histogram_intersection:
    def __init__(self) -> None:
        ...
    def train(self, arg0: sparse_vectors, arg1: array) -> _decision_function_sparse_histogram_intersection:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class rvm_trainer_sparse_linear:
    def __init__(self) -> None:
        ...
    def train(self, arg0: sparse_vectors, arg1: array) -> _decision_function_sparse_linear:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class rvm_trainer_sparse_radial_basis:
    def __init__(self) -> None:
        ...
    def train(self, arg0: sparse_vectors, arg1: array) -> _decision_function_sparse_radial_basis:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gamma(self) -> float:
        ...
    @gamma.setter
    def gamma(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class segmenter_params:
    """
    This class is used to define all the optional parameters to the    
    train_sequence_segmenter() and cross_validate_sequence_segmenter() routines.   
    """
    allow_negative_weights: bool
    be_verbose: bool
    use_BIO_model: bool
    use_high_order_features: bool
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def C(self) -> float:
        """
        SVM C parameter
        """
    @C.setter
    def C(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_cache_size(self) -> int:
        ...
    @max_cache_size.setter
    def max_cache_size(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def num_threads(self) -> int:
        ...
    @num_threads.setter
    def num_threads(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def window_size(self) -> int:
        ...
    @window_size.setter
    def window_size(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class segmenter_test:
    """
    This object is the output of the dlib.test_sequence_segmenter() and dlib.cross_validate_sequence_segmenter() routines.
    """
    def __getstate__(self) -> tuple:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def f1(self) -> float:
        ...
    @f1.setter
    def f1(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def precision(self) -> float:
        ...
    @precision.setter
    def precision(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def recall(self) -> float:
        ...
    @recall.setter
    def recall(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class segmenter_type:
    """
    This object represents a sequence segmenter and is the type of object returned by the dlib.train_sequence_segmenter() routine.
    """
    @typing.overload
    def __call__(self, arg0: vectors) -> ranges:
        ...
    @typing.overload
    def __call__(self, arg0: sparse_vectors) -> ranges:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def weights(self) -> vector:
        ...
class shape_predictor:
    """
    This object is a tool that takes in an image region containing some object and outputs a set of point locations that define the pose of the object. The classic example of this is human face pose prediction, where you take an image of a human face as input and are expected to identify the locations of important facial landmarks such as the corners of the mouth and eyes, tip of the nose, and so forth.
    """
    def __call__(self, image: numpy.ndarray, box: rectangle) -> full_object_detection:
        """
        requires 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
            - box is the bounding box to begin the shape prediction inside. 
        ensures 
            - This function runs the shape predictor on the input image and returns 
              a single full_object_detection.
        """
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        """
        Loads a shape_predictor from a file that contains the output of the 
        train_shape_predictor() routine.
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def save(self, predictor_output_filename: str) -> None:
        """
        Save a shape_predictor to the provided path.
        """
class shape_predictor_training_options:
    """
    This object is a container for the options to the train_shape_predictor() routine.
    """
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def be_verbose(self) -> bool:
        """
        If true, train_shape_predictor() will print out a lot of information to stdout while training.
        """
    @be_verbose.setter
    def be_verbose(self, arg0: bool) -> None:
        ...
    @property
    def cascade_depth(self) -> int:
        """
        The number of cascades created to train the model with.
        """
    @cascade_depth.setter
    def cascade_depth(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def feature_pool_region_padding(self) -> float:
        """
        Size of region within which to sample features for the feature pool.                       positive values increase the sampling region while negative values decrease it. E.g. padding of 0 means we                       sample fr
        """
    @feature_pool_region_padding.setter
    def feature_pool_region_padding(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def feature_pool_size(self) -> int:
        """
        Number of pixels used to generate features for the random trees.
        """
    @feature_pool_size.setter
    def feature_pool_size(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def lambda_param(self) -> float:
        """
        Controls how tight the feature sampling should be. Lower values enforce closer features.
        """
    @lambda_param.setter
    def lambda_param(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def landmark_relative_padding_mode(self) -> bool:
        """
        If True then features are drawn only from the box around the landmarks, otherwise they come from the bounding box and landmarks together.  See feature_pool_region_padding doc for more details.
        """
    @landmark_relative_padding_mode.setter
    def landmark_relative_padding_mode(self, arg0: bool) -> None:
        ...
    @property
    def nu(self) -> float:
        """
        The regularization parameter.  Larger values of this parameter                        will cause the algorithm to fit the training data better but may also                        cause overfitting.  The value must be in the range (0, 1].
        """
    @nu.setter
    def nu(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def num_test_splits(self) -> int:
        """
        Number of split features at each node to sample. The one that gives the best split is chosen.
        """
    @num_test_splits.setter
    def num_test_splits(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def num_threads(self) -> int:
        """
        Use this many threads/CPU cores for training.
        """
    @num_threads.setter
    def num_threads(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def num_trees_per_cascade_level(self) -> int:
        """
        The number of trees created for each cascade.
        """
    @num_trees_per_cascade_level.setter
    def num_trees_per_cascade_level(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def oversampling_amount(self) -> int:
        """
        The number of randomly selected initial starting points sampled for each training example
        """
    @oversampling_amount.setter
    def oversampling_amount(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def oversampling_translation_jitter(self) -> float:
        """
        The amount of translation jittering to apply to bounding boxes, a good value is in in the range [0 0.5].
        """
    @oversampling_translation_jitter.setter
    def oversampling_translation_jitter(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def random_seed(self) -> str:
        """
        The random seed used by the internal random number generator
        """
    @random_seed.setter
    def random_seed(self, arg0: str) -> None:
        ...
    @property
    def tree_depth(self) -> int:
        """
        The depth of the trees used in each cascade. There are pow(2, get_tree_depth()) leaves in each tree
        """
    @tree_depth.setter
    def tree_depth(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class simple_object_detector:
    """
    This object represents a sliding window histogram-of-oriented-gradients based object detector.
    """
    @staticmethod
    def run_multiple(detectors: list, image: numpy.ndarray, upsample_num_times: typing.SupportsInt | typing.SupportsIndex = 0, adjust_threshold: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> tuple:
        """
        requires 
            - detectors is a list of detectors. 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
            - upsample_num_times >= 0 
        ensures 
            - This function runs the list of object detectors at once on the input image and returns 
              a tuple of (list of detections, list of scores, list of weight_indices).   
            - Upsamples the image upsample_num_times before running the basic 
              detector.
        """
    @typing.overload
    def __call__(self, image: numpy.ndarray, upsample_num_times: typing.SupportsInt | typing.SupportsIndex) -> rectangles:
        """
        requires 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
            - upsample_num_times >= 0 
        ensures 
            - This function runs the object detector on the input image and returns 
              a list of detections.   
            - Upsamples the image upsample_num_times before running the basic 
              detector.  If you don't know how many times you want to upsample then 
              don't provide a value for upsample_num_times and an appropriate 
              default will be used.
        """
    @typing.overload
    def __call__(self, image: numpy.ndarray) -> rectangles:
        """
        requires 
            - image is a numpy ndarray containing either an 8bit grayscale or RGB 
              image. 
        ensures 
            - This function runs the object detector on the input image and returns 
              a list of detections.
        """
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self, detectors: list) -> None:
        """
        This version of the constructor builds a simple_object_detector from a 
        bunch of other simple_object_detectors.  It essentially packs them together 
        so that when you run the detector it's like calling run_multiple().  Except 
        in this case the non-max suppression is applied to them all as a group.  So 
        unlike run_multiple(), each detector competes in the non-max suppression. 
         
        Also, the non-max suppression settings used for this whole thing are 
        the settings used by detectors[0].  So if you have a preference,  
        put the detector that uses the type of non-max suppression you like first 
        in the list.
        """
    @typing.overload
    def __init__(self, arg0: str) -> None:
        """
        Loads a simple_object_detector from a file that contains the output of the 
        train_simple_object_detector() routine.
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def save(self, detector_output_filename: str) -> None:
        """
        Save a simple_object_detector to the provided path.
        """
    @property
    def detection_window_height(self) -> int:
        ...
    @property
    def detection_window_width(self) -> int:
        ...
    @property
    def num_detectors(self) -> int:
        ...
    @property
    def upsampling_amount(self) -> int:
        """
        The detector upsamples the image this many times before running.
        """
    @upsampling_amount.setter
    def upsampling_amount(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class simple_object_detector_training_options:
    """
    This object is a container for the options to the train_simple_object_detector() routine.
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def C(self) -> float:
        """
        C is the usual SVM C regularization parameter.  So it is passed to 
        structural_object_detection_trainer::set_c().  Larger values of C 
        will encourage the trainer to fit the data better but might lead to 
        overfitting.  Therefore, you must determine the proper setting of 
        this parameter experimentally.
        """
    @C.setter
    def C(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def add_left_right_image_flips(self) -> bool:
        """
        if true, train_simple_object_detector() will assume the objects are 
        left/right symmetric and add in left right flips of the training 
        images.  This doubles the size of the training dataset.
        """
    @add_left_right_image_flips.setter
    def add_left_right_image_flips(self, arg0: bool) -> None:
        ...
    @property
    def be_verbose(self) -> bool:
        """
        If true, train_simple_object_detector() will print out a lot of information to the screen while training.
        """
    @be_verbose.setter
    def be_verbose(self, arg0: bool) -> None:
        ...
    @property
    def detection_window_size(self) -> int:
        """
        The sliding window used will have about this many pixels inside it.
        """
    @detection_window_size.setter
    def detection_window_size(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def epsilon(self) -> float:
        """
        epsilon is the stopping epsilon.  Smaller values make the trainer's 
        solver more accurate but might take longer to train.
        """
    @epsilon.setter
    def epsilon(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_runtime_seconds(self) -> float:
        """
        Don't let the solver run for longer than this many seconds.
        """
    @max_runtime_seconds.setter
    def max_runtime_seconds(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def nuclear_norm_regularization_strength(self) -> float:
        """
        This detector works by convolving a filter over a HOG feature image.  If that 
        filter is separable then the convolution can be performed much faster.  The 
        nuclear_norm_regularization_strength parameter encourages the machine learning 
        algorithm to learn a separable filter.  A value of 0 disables this feature, but 
        any non-zero value places a nuclear norm regularizer on the objective function 
        and this encourages the learning of a separable filter.  Note that setting 
        nuclear_norm_regularization_strength to a non-zero value can make the training 
        process take significantly longer, so be patient when using it.
        """
    @nuclear_norm_regularization_strength.setter
    def nuclear_norm_regularization_strength(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def num_threads(self) -> int:
        """
        train_simple_object_detector() will use this many threads of 
        execution.  Set this to the number of CPU cores on your machine to 
        obtain the fastest training speed.
        """
    @num_threads.setter
    def num_threads(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def upsample_limit(self) -> int:
        """
        train_simple_object_detector() will upsample images if needed 
        no more than upsample_limit times. Value 0 will forbid trainer to 
        upsample any images. If trainer is unable to fit all boxes with 
        required upsample_limit, exception will be thrown. Higher values 
        of upsample_limit exponentially increases memory requirements. 
        Values higher than 2 (default) are not recommended.
        """
    @upsample_limit.setter
    def upsample_limit(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class simple_test_results:
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def average_precision(self) -> float:
        ...
    @average_precision.setter
    def average_precision(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def precision(self) -> float:
        ...
    @precision.setter
    def precision(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def recall(self) -> float:
        ...
    @recall.setter
    def recall(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class sparse_ranking_pair:
    nonrelevant: sparse_vectors
    relevant: sparse_vectors
    def __getstate__(self) -> tuple:
        ...
    def __init__(self) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
class sparse_ranking_pairs:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: sparse_ranking_pair) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: sparse_ranking_pairs) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> sparse_ranking_pairs:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> sparse_ranking_pair:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: sparse_ranking_pairs) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[sparse_ranking_pair]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: sparse_ranking_pairs) -> bool:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: sparse_ranking_pair) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: sparse_ranking_pairs) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: sparse_ranking_pair) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: sparse_ranking_pair) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: sparse_ranking_pairs) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: sparse_ranking_pair) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> sparse_ranking_pair:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> sparse_ranking_pair:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: sparse_ranking_pair) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class sparse_vector:
    """
    This object represents the mathematical idea of a sparse column vector.  It is    
    simply an array of dlib.pair objects, each representing an index/value pair in    
    the vector.  Any elements of the vector which are missing are implicitly set to    
    zero.      
        
    Unless otherwise noted, any routines taking a sparse_vector assume the sparse    
    vector is sorted and has unique elements.  That is, the index values of the    
    pairs in a sparse_vector should be listed in increasing order and there should    
    not be duplicates.  However, some functions work with "unsorted" sparse    
    vectors.  These are dlib.sparse_vector objects that have either duplicate    
    entries or non-sorted index values.  Note further that you can convert an    
    "unsorted" sparse_vector into a properly sorted sparse vector by calling    
    dlib.make_sparse_vector() on it.   
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: pair) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: sparse_vector) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> sparse_vector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> pair:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: sparse_vector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[pair]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: sparse_vector) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: pair) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: sparse_vector) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    def append(self, x: pair) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: pair) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: sparse_vector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: pair) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> pair:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> pair:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: pair) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class sparse_vectors:
    """
    This object is an array of sparse_vector objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: sparse_vector) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: sparse_vectors) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> sparse_vectors:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> sparse_vector:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: sparse_vectors) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[sparse_vector]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: sparse_vectors) -> bool:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: sparse_vector) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: sparse_vectors) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: sparse_vector) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: sparse_vector) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: sparse_vectors) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: sparse_vector) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> sparse_vector:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> sparse_vector:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: sparse_vector) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class sparse_vectorss:
    """
    This object is an array of arrays of sparse_vector objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: sparse_vectors) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: sparse_vectorss) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> sparse_vectorss:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> sparse_vectors:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: sparse_vectorss) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[sparse_vectors]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: sparse_vectorss) -> bool:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: sparse_vectors) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: sparse_vectorss) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: sparse_vectors) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: sparse_vectors) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: sparse_vectorss) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: sparse_vectors) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> sparse_vectors:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> sparse_vectors:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: sparse_vectors) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class svm_c_trainer_histogram_intersection:
    def __init__(self) -> None:
        ...
    def set_c(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def train(self, arg0: vectors, arg1: array) -> _decision_function_histogram_intersection:
        ...
    @property
    def c_class1(self) -> float:
        ...
    @c_class1.setter
    def c_class1(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def c_class2(self) -> float:
        ...
    @c_class2.setter
    def c_class2(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cache_size(self) -> int:
        ...
    @cache_size.setter
    def cache_size(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class svm_c_trainer_linear:
    force_last_weight_to_1: bool
    learns_nonnegative_weights: bool
    def __init__(self) -> None:
        ...
    def be_quiet(self) -> None:
        ...
    def be_verbose(self) -> None:
        ...
    def set_c(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_prior(self, arg0: _decision_function_linear) -> None:
        ...
    def train(self, arg0: vectors, arg1: array) -> _decision_function_linear:
        ...
    @property
    def c_class1(self) -> float:
        ...
    @c_class1.setter
    def c_class1(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def c_class2(self) -> float:
        ...
    @c_class2.setter
    def c_class2(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def has_prior(self) -> bool:
        ...
    @property
    def max_iterations(self) -> int:
        ...
    @max_iterations.setter
    def max_iterations(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class svm_c_trainer_radial_basis:
    def __init__(self) -> None:
        ...
    def set_c(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def train(self, arg0: vectors, arg1: array) -> _decision_function_radial_basis:
        ...
    @property
    def c_class1(self) -> float:
        ...
    @c_class1.setter
    def c_class1(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def c_class2(self) -> float:
        ...
    @c_class2.setter
    def c_class2(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cache_size(self) -> int:
        ...
    @cache_size.setter
    def cache_size(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gamma(self) -> float:
        ...
    @gamma.setter
    def gamma(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class svm_c_trainer_sparse_histogram_intersection:
    def __init__(self) -> None:
        ...
    def set_c(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def train(self, arg0: sparse_vectors, arg1: array) -> _decision_function_sparse_histogram_intersection:
        ...
    @property
    def c_class1(self) -> float:
        ...
    @c_class1.setter
    def c_class1(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def c_class2(self) -> float:
        ...
    @c_class2.setter
    def c_class2(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cache_size(self) -> int:
        ...
    @cache_size.setter
    def cache_size(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class svm_c_trainer_sparse_linear:
    force_last_weight_to_1: bool
    learns_nonnegative_weights: bool
    def __init__(self) -> None:
        ...
    def be_quiet(self) -> None:
        ...
    def be_verbose(self) -> None:
        ...
    def set_c(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_prior(self, arg0: _decision_function_sparse_linear) -> None:
        ...
    def train(self, arg0: sparse_vectors, arg1: array) -> _decision_function_sparse_linear:
        ...
    @property
    def c_class1(self) -> float:
        ...
    @c_class1.setter
    def c_class1(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def c_class2(self) -> float:
        ...
    @c_class2.setter
    def c_class2(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def has_prior(self) -> bool:
        ...
    @property
    def max_iterations(self) -> int:
        ...
    @max_iterations.setter
    def max_iterations(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class svm_c_trainer_sparse_radial_basis:
    def __init__(self) -> None:
        ...
    def set_c(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def train(self, arg0: sparse_vectors, arg1: array) -> _decision_function_sparse_radial_basis:
        ...
    @property
    def c_class1(self) -> float:
        ...
    @c_class1.setter
    def c_class1(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def c_class2(self) -> float:
        ...
    @c_class2.setter
    def c_class2(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cache_size(self) -> int:
        ...
    @cache_size.setter
    def cache_size(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gamma(self) -> float:
        ...
    @gamma.setter
    def gamma(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class svm_rank_trainer:
    force_last_weight_to_1: bool
    learns_nonnegative_weights: bool
    def __init__(self) -> None:
        ...
    def be_quiet(self) -> None:
        ...
    def be_verbose(self) -> None:
        ...
    def set_prior(self, arg0: ..., 0l, 1l, dlib: ..., dlib: ...) -> None:
        ...
    @typing.overload
    def train(self, arg0: ranking_pair) -> ...:
        ...
    @typing.overload
    def train(self, arg0: ranking_pairs) -> ...:
        ...
    @property
    def c(self) -> float:
        ...
    @c.setter
    def c(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def has_prior(self) -> bool:
        ...
    @property
    def max_iterations(self) -> int:
        ...
    @max_iterations.setter
    def max_iterations(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class svm_rank_trainer_sparse:
    force_last_weight_to_1: bool
    learns_nonnegative_weights: bool
    @staticmethod
    def set_prior(*args, **kwargs) -> None:
        ...
    def __init__(self) -> None:
        ...
    def be_quiet(self) -> None:
        ...
    def be_verbose(self) -> None:
        ...
    @typing.overload
    def train(self, arg0: sparse_ranking_pair) -> ...:
        ...
    @typing.overload
    def train(self, arg0: sparse_ranking_pairs) -> ...:
        ...
    @property
    def c(self) -> float:
        ...
    @c.setter
    def c(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def has_prior(self) -> bool:
        ...
    @property
    def max_iterations(self) -> int:
        ...
    @max_iterations.setter
    def max_iterations(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class vector:
    """
    This object represents the mathematical idea of a column vector.
    """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> float:
        ...
    @typing.overload
    def __getitem__(self, arg0: slice) -> vector:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing.Any) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def __str__(self) -> str:
        ...
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def set_size(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def shape(self) -> tuple:
        ...
class vectors:
    """
    This object is an array of vector objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: vector) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: vectors) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> vectors:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> vector:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: vectors) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[vector]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: vectors) -> bool:
        ...
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: vector) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: vectors) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: vector) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: vector) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: vectors) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: vector) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> vector:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> vector:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: vector) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class vectorss:
    """
    This object is an array of arrays of vector objects.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: vectors) -> bool:
        """
        Return true the container contains ``x``
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
    def __eq__(self, arg0: vectorss) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> vectorss:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> vectors:
        ...
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: vectorss) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[vectors]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: vectorss) -> bool:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: vectors) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: vectorss) -> None:
        """
        Assign list elements using a slice object
        """
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, x: vectors) -> None:
        """
        Add an item to the end of the list
        """
    @typing.overload
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def clear(self) -> None:
        ...
    def count(self, x: vectors) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: vectorss) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, arg0: list) -> None:
        ...
    def insert(self, i: typing.SupportsInt | typing.SupportsIndex, x: vectors) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> vectors:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt | typing.SupportsIndex) -> vectors:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: vectors) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    def resize(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
def angle_between_lines(a: line, b: line) -> float:
    """
    ensures 
        - returns the angle, in degrees, between the given lines.  This is a number in 
          the range [0 90].
    """
def apply_cca_transform(m: matrix, v: sparse_vector) -> vector:
    """
    requires    
        - max_index_plus_one(v) <= m.nr()    
    ensures    
        - returns trans(m)*v    
          (i.e. multiply m by the vector v and return the result)   
    """
def as_grayscale(img: numpy.ndarray) -> numpy.ndarray:
    """
    Convert an image to 8bit grayscale.  If it's already a grayscale image do nothing and just return img.
    """
def assignment_cost(cost: matrix, assignment: list) -> float:
    """
    requires    
        - cost.nr() == cost.nc()    
          (i.e. the input must be a square matrix)    
        - for all valid i:    
            - 0 <= assignment[i] < cost.nr()    
    ensures    
        - Interprets cost as a cost assignment matrix. That is, cost[i][j]     
          represents the cost of assigning i to j.      
        - Interprets assignment as a particular set of assignments. That is,    
          i is assigned to assignment[i].    
        - returns the cost of the given assignment. That is, returns    
          a number which is:    
            sum over i: cost[i][assignment[i]]   
    """
@typing.overload
def auto_train_rbf_classifier(x: vectors, y: array, max_runtime_seconds: typing.SupportsFloat | typing.SupportsIndex, be_verbose: bool = True) -> _normalized_decision_function_radial_basis:
    """
    requires 
        - y contains at least 6 examples of each class.  Moreover, every element in y 
          is either +1 or -1. 
        - max_runtime_seconds >= 0 
        - len(x) == len(y) 
        - all the vectors in x have the same dimension. 
    ensures 
        - This routine trains a radial basis function SVM on the given binary 
          classification training data.  It uses the svm_c_trainer to do this.  It also 
          uses find_max_global() and 6-fold cross-validation to automatically determine 
          the best settings of the SVM's hyper parameters. 
        - Note that we interpret y[i] as the label for the vector x[i].  Therefore, the 
          returned function, df, should generally satisfy sign(df(x[i])) == y[i] as 
          often as possible. 
        - The hyperparameter search will run for about max_runtime and will print 
          messages to the screen as it runs if be_verbose==true.
    """
@typing.overload
def auto_train_rbf_classifier(x: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], y: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], max_runtime_seconds: typing.SupportsFloat | typing.SupportsIndex, be_verbose: bool = True) -> _normalized_decision_function_radial_basis:
    """
    requires 
        - y contains at least 6 examples of each class.  Moreover, every element in y 
          is either +1 or -1. 
        - max_runtime_seconds >= 0 
        - len(x.shape(0)) == len(y) 
        - x.shape(1) > 0 
    ensures 
        - This routine trains a radial basis function SVM on the given binary 
          classification training data.  It uses the svm_c_trainer to do this.  It also 
          uses find_max_global() and 6-fold cross-validation to automatically determine 
          the best settings of the SVM's hyper parameters. 
        - Note that we interpret y[i] as the label for the vector x[i].  Therefore, the 
          returned function, df, should generally satisfy sign(df(x[i])) == y[i] as 
          often as possible. 
        - The hyperparameter search will run for about max_runtime and will print 
          messages to the screen as it runs if be_verbose==true.
    """
def bottom_up_clustering(descriptors: list, min_num_clusters: typing.SupportsInt | typing.SupportsIndex = 1, max_dist: typing.SupportsFloat | typing.SupportsIndex = 0.6) -> list:
    """
    Takes a list of descriptors and returns a list that contains a label for each descriptor. Clustering is done using dlib::bottom_up_cluster.
    """
def cca(L: sparse_vectors, R: sparse_vectors, num_correlations: typing.SupportsInt | typing.SupportsIndex, extra_rank: typing.SupportsInt | typing.SupportsIndex = 5, q: typing.SupportsInt | typing.SupportsIndex = 2, regularization: typing.SupportsFloat | typing.SupportsIndex = 0) -> cca_outputs:
    """
    requires    
        - num_correlations > 0    
        - len(L) > 0     
        - len(R) > 0     
        - len(L) == len(R)    
        - regularization >= 0    
        - L and R must be properly sorted sparse vectors.  This means they must list their  
          elements in ascending index order and not contain duplicate index values.  You can use 
          make_sparse_vector() to ensure this is true.  
    ensures    
        - This function performs a canonical correlation analysis between the vectors    
          in L and R.  That is, it finds two transformation matrices, Ltrans and    
          Rtrans, such that row vectors in the transformed matrices L*Ltrans and    
          R*Rtrans are as correlated as possible (note that in this notation we    
          interpret L as a matrix with the input vectors in its rows).  Note also that    
          this function tries to find transformations which produce num_correlations    
          dimensional output vectors.    
        - Note that you can easily apply the transformation to a vector using     
          apply_cca_transform().  So for example, like this:     
            - apply_cca_transform(Ltrans, some_sparse_vector)    
        - returns a structure containing the Ltrans and Rtrans transformation matrices    
          as well as the estimated correlations between elements of the transformed    
          vectors.    
        - This function assumes the data vectors in L and R have already been centered    
          (i.e. we assume the vectors have zero means).  However, in many cases it is    
          fine to use uncentered data with cca().  But if it is important for your    
          problem then you should center your data before passing it to cca().   
        - This function works with reduced rank approximations of the L and R matrices.    
          This makes it fast when working with large matrices.  In particular, we use    
          the dlib::svd_fast() routine to find reduced rank representations of the input    
          matrices by calling it as follows: svd_fast(L, U,D,V, num_correlations+extra_rank, q)     
          and similarly for R.  This means that you can use the extra_rank and q    
          arguments to cca() to influence the accuracy of the reduced rank    
          approximation.  However, the default values should work fine for most    
          problems.    
        - The dimensions of the output vectors produced by L*#Ltrans or R*#Rtrans are 
          ordered such that the dimensions with the highest correlations come first. 
          That is, after applying the transforms produced by cca() to a set of vectors 
          you will find that dimension 0 has the highest correlation, then dimension 1 
          has the next highest, and so on.  This also means that the list of estimated 
          correlations returned from cca() will always be listed in decreasing order. 
        - This function performs the ridge regression version of Canonical Correlation    
          Analysis when regularization is set to a value > 0.  In particular, larger    
          values indicate the solution should be more heavily regularized.  This can be    
          useful when the dimensionality of the data is larger than the number of    
          samples.    
        - A good discussion of CCA can be found in the paper "Canonical Correlation    
          Analysis" by David Weenink.  In particular, this function is implemented    
          using equations 29 and 30 from his paper.  We also use the idea of doing CCA    
          on a reduced rank approximation of L and R as suggested by Paramveer S.    
          Dhillon in his paper "Two Step CCA: A new spectral method for estimating    
          vector models of words".   
    """
@typing.overload
def center(rect: rectangle) -> point:
    """
        returns the center of the given rectangle
    """
@typing.overload
def center(rect: drectangle) -> dpoint:
    """
        returns the center of the given rectangle
    """
@typing.overload
def centered_rect(p: point, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> rectangle:
    ...
@typing.overload
def centered_rect(p: dpoint, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> rectangle:
    ...
@typing.overload
def centered_rect(rect: rectangle, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> rectangle:
    ...
@typing.overload
def centered_rect(rect: drectangle, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> rectangle:
    ...
def centered_rects(pts: points, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> rectangles:
    ...
def chinese_whispers(edges: list) -> list:
    """
    Given a graph with vertices represented as numbers indexed from 0, this algorithm takes a list of edges and returns back a list that contains a labels (found clusters) for each vertex. Edges are tuples with either 2 elements (integers presenting indexes of connected vertices) or 3 elements, where additional one element is float which presents distance weight of the edge). Offers direct access to dlib::chinese_whispers.
    """
def chinese_whispers_clustering(descriptors: list, threshold: typing.SupportsFloat | typing.SupportsIndex) -> list:
    """
    Takes a list of descriptors and returns a list that contains a label for each descriptor. Clustering is done using dlib::chinese_whispers.
    """
@typing.overload
def convert_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], dtype: str) -> numpy.ndarray:
    ...
@typing.overload
def convert_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], dtype: str) -> numpy.ndarray:
    ...
@typing.overload
def convert_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], dtype: str) -> numpy.ndarray:
    ...
@typing.overload
def convert_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], dtype: str) -> numpy.ndarray:
    ...
@typing.overload
def convert_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], dtype: str) -> numpy.ndarray:
    ...
@typing.overload
def convert_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], dtype: str) -> numpy.ndarray:
    ...
@typing.overload
def convert_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], dtype: str) -> numpy.ndarray:
    ...
@typing.overload
def convert_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], dtype: str) -> numpy.ndarray:
    ...
@typing.overload
def convert_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], dtype: str) -> numpy.ndarray:
    ...
@typing.overload
def convert_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], dtype: str) -> numpy.ndarray:
    ...
@typing.overload
def convert_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], dtype: str) -> numpy.ndarray:
    """
    Converts an image to a target pixel type.  dtype must be a string containing one of the following: 
        uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float, float64, double, or rgb_pixel 
     
    When converting from a color space with more than 255 values the pixel intensity is 
    saturated at the minimum and maximum pixel values of the target pixel type.  For 
    example, if you convert a float valued image to uint8 then float values will be 
    truncated to integers and values larger than 255 are converted to 255 while values less 
    than 0 are converted to 0.
    """
@typing.overload
def convert_image_scaled(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], dtype: str, thresh: typing.SupportsFloat | typing.SupportsIndex = 4) -> numpy.ndarray:
    ...
@typing.overload
def convert_image_scaled(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], dtype: str, thresh: typing.SupportsFloat | typing.SupportsIndex = 4) -> numpy.ndarray:
    ...
@typing.overload
def convert_image_scaled(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], dtype: str, thresh: typing.SupportsFloat | typing.SupportsIndex = 4) -> numpy.ndarray:
    ...
@typing.overload
def convert_image_scaled(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], dtype: str, thresh: typing.SupportsFloat | typing.SupportsIndex = 4) -> numpy.ndarray:
    ...
@typing.overload
def convert_image_scaled(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], dtype: str, thresh: typing.SupportsFloat | typing.SupportsIndex = 4) -> numpy.ndarray:
    ...
@typing.overload
def convert_image_scaled(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], dtype: str, thresh: typing.SupportsFloat | typing.SupportsIndex = 4) -> numpy.ndarray:
    ...
@typing.overload
def convert_image_scaled(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], dtype: str, thresh: typing.SupportsFloat | typing.SupportsIndex = 4) -> numpy.ndarray:
    ...
@typing.overload
def convert_image_scaled(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], dtype: str, thresh: typing.SupportsFloat | typing.SupportsIndex = 4) -> numpy.ndarray:
    ...
@typing.overload
def convert_image_scaled(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], dtype: str, thresh: typing.SupportsFloat | typing.SupportsIndex = 4) -> numpy.ndarray:
    ...
@typing.overload
def convert_image_scaled(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], dtype: str, thresh: typing.SupportsFloat | typing.SupportsIndex = 4) -> numpy.ndarray:
    ...
@typing.overload
def convert_image_scaled(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], dtype: str, thresh: typing.SupportsFloat | typing.SupportsIndex = 4) -> numpy.ndarray:
    """
    requires 
        - thresh > 0 
    ensures 
        - Converts an image to a target pixel type.  dtype must be a string containing one of the following: 
          uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float, float64, double, or rgb_pixel 
     
          The contents of img will be scaled to fit the dynamic range of the target 
          pixel type.  The thresh parameter is used to filter source pixel values which 
          are outliers.  These outliers will saturate at the edge of the destination 
          image's dynamic range. 
        - Specifically, for all valid r and c: 
            - We scale img[r][c] into the dynamic range of the target pixel type.  This 
              is done using the mean and standard deviation of img. Call the mean M and 
              the standard deviation D.  Then the scaling from source to destination is 
              performed using the following mapping: 
                let SRC_UPPER  = min(M + thresh*D, max(img)) 
                let SRC_LOWER  = max(M - thresh*D, min(img)) 
                let DEST_UPPER = max value possible for the selected dtype.  
                let DEST_LOWER = min value possible for the selected dtype. 
     
                MAPPING: [SRC_LOWER, SRC_UPPER] -> [DEST_LOWER, DEST_UPPER] 
     
              Where this mapping is a linear mapping of values from the left range 
              into the right range of values.  Source pixel values outside the left 
              range are modified to be at the appropriate end of the range.
    """
@typing.overload
def count_points_between_lines(l1: line, l2: line, reference_point: dpoint, pts: points) -> float:
    ...
@typing.overload
def count_points_between_lines(l1: line, l2: line, reference_point: dpoint, pts: dpoints) -> float:
    """
    ensures 
        - Counts and returns the number of points in pts that are between lines l1 and 
          l2.  Since a pair of lines will, in the general case, divide the plane into 4 
          regions, we identify the region of interest as the one that contains the 
          reference_point.  Therefore, this function counts the number of points in pts 
          that appear in the same region as reference_point.
    """
@typing.overload
def count_points_on_side_of_line(l: line, reference_point: dpoint, pts: points, dist_thresh_min: typing.SupportsFloat | typing.SupportsIndex = 0, dist_thresh_max: typing.SupportsFloat | typing.SupportsIndex = ...) -> int:
    ...
@typing.overload
def count_points_on_side_of_line(l: line, reference_point: dpoint, pts: dpoints, dist_thresh_min: typing.SupportsFloat | typing.SupportsIndex = 0, dist_thresh_max: typing.SupportsFloat | typing.SupportsIndex = ...) -> int:
    """
    ensures 
        - Returns a count of how many points in pts have a distance from the line l 
          that is in the range [dist_thresh_min, dist_thresh_max].  This distance is a 
          signed value that indicates how far a point is from the line. Moreover, if 
          the point is on the same side as reference_point then the distance is 
          positive, otherwise it is negative.  So for example, If this range is [0, 
          infinity] then this function counts how many points are on the same side of l 
          as reference_point.
    """
def count_steps_without_decrease(time_series: typing.Any, probability_of_decrease: typing.SupportsFloat | typing.SupportsIndex = 0.51) -> int:
    """
    requires 
        - time_series must be a one dimensional array of real numbers.  
        - 0.5 < probability_of_decrease < 1 
    ensures 
        - If you think of the contents of time_series as a potentially noisy time 
          series, then this function returns a count of how long the time series has 
          gone without noticeably decreasing in value.  It does this by scanning along 
          the elements, starting from the end (i.e. time_series[-1]) to the beginning, 
          and checking how many elements you need to examine before you are confident 
          that the series has been decreasing in value.  Here, "confident of decrease" 
          means the probability of decrease is >= probability_of_decrease.   
        - Setting probability_of_decrease to 0.51 means we count until we see even a 
          small hint of decrease, whereas a larger value of 0.99 would return a larger 
          count since it keeps going until it is nearly certain the time series is 
          decreasing. 
        - The max possible output from this function is len(time_series). 
        - The implementation of this function is done using the dlib::running_gradient 
          object, which is a tool that finds the least squares fit of a line to the 
          time series and the confidence interval around the slope of that line.  That 
          can then be used in a simple statistical test to determine if the slope is 
          positive or negative.
    """
def count_steps_without_decrease_robust(time_series: typing.Any, probability_of_decrease: typing.SupportsFloat | typing.SupportsIndex = 0.51, quantile_discard: typing.SupportsFloat | typing.SupportsIndex = 0.1) -> int:
    """
    requires 
        - time_series must be a one dimensional array of real numbers.  
        - 0.5 < probability_of_decrease < 1 
        - 0 <= quantile_discard <= 1 
    ensures 
        - This function behaves just like 
          count_steps_without_decrease(time_series,probability_of_decrease) except that 
          it ignores values in the time series that are in the upper quantile_discard 
          quantile.  So for example, if the quantile discard is 0.1 then the 10% 
          largest values in the time series are ignored.
    """
@typing.overload
def cross_validate_ranking_trainer(trainer: svm_rank_trainer, samples: ranking_pairs, folds: typing.SupportsInt | typing.SupportsIndex) -> ranking_test:
    ...
@typing.overload
def cross_validate_ranking_trainer(trainer: svm_rank_trainer_sparse, samples: sparse_ranking_pairs, folds: typing.SupportsInt | typing.SupportsIndex) -> ranking_test:
    ...
@typing.overload
def cross_validate_sequence_segmenter(*args, **kwargs) -> segmenter_test:
    ...
@typing.overload
def cross_validate_sequence_segmenter(*args, **kwargs) -> segmenter_test:
    ...
@typing.overload
def cross_validate_trainer(trainer: svm_c_trainer_radial_basis, x: vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer(trainer: svm_c_trainer_sparse_radial_basis, x: sparse_vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer(trainer: svm_c_trainer_histogram_intersection, x: vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer(trainer: svm_c_trainer_sparse_histogram_intersection, x: sparse_vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer(trainer: svm_c_trainer_linear, x: vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer(trainer: svm_c_trainer_sparse_linear, x: sparse_vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer(trainer: rvm_trainer_radial_basis, x: vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer(trainer: rvm_trainer_sparse_radial_basis, x: sparse_vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer(trainer: rvm_trainer_histogram_intersection, x: vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer(trainer: rvm_trainer_sparse_histogram_intersection, x: sparse_vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer(trainer: rvm_trainer_linear, x: vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer(trainer: rvm_trainer_sparse_linear, x: sparse_vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer_threaded(trainer: svm_c_trainer_radial_basis, x: vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex, num_threads: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer_threaded(trainer: svm_c_trainer_sparse_radial_basis, x: sparse_vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex, num_threads: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer_threaded(trainer: svm_c_trainer_histogram_intersection, x: vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex, num_threads: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer_threaded(trainer: svm_c_trainer_sparse_histogram_intersection, x: sparse_vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex, num_threads: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer_threaded(trainer: svm_c_trainer_linear, x: vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex, num_threads: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer_threaded(trainer: svm_c_trainer_sparse_linear, x: sparse_vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex, num_threads: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer_threaded(trainer: rvm_trainer_radial_basis, x: vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex, num_threads: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer_threaded(trainer: rvm_trainer_sparse_radial_basis, x: sparse_vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex, num_threads: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer_threaded(trainer: rvm_trainer_histogram_intersection, x: vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex, num_threads: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer_threaded(trainer: rvm_trainer_sparse_histogram_intersection, x: sparse_vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex, num_threads: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer_threaded(trainer: rvm_trainer_linear, x: vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex, num_threads: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def cross_validate_trainer_threaded(trainer: rvm_trainer_sparse_linear, x: sparse_vectors, y: array, folds: typing.SupportsInt | typing.SupportsIndex, num_threads: typing.SupportsInt | typing.SupportsIndex) -> _binary_test:
    ...
@typing.overload
def distance_to_line(l: line, p: point) -> float:
    ...
@typing.overload
def distance_to_line(l: line, p: dpoint) -> float:
    """
    returns abs(signed_distance_to_line(l,p))
    """
@typing.overload
def dot(arg0: vector, arg1: vector) -> float:
    """
    Compute the dot product between two dense column vectors.
    """
@typing.overload
def dot(a: point, b: point) -> int:
    """
    Returns the dot product of the points a and b.
    """
@typing.overload
def dot(a: dpoint, b: dpoint) -> float:
    """
    Returns the dot product of the points a and b.
    """
@typing.overload
def equalize_histogram(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def equalize_histogram(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]:
    """
    Returns a histogram equalized version of img.
    """
@typing.overload
def extract_image_4points(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], corners: list, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def extract_image_4points(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], corners: list, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]:
    ...
@typing.overload
def extract_image_4points(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], corners: list, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]:
    ...
@typing.overload
def extract_image_4points(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], corners: list, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]:
    ...
@typing.overload
def extract_image_4points(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], corners: list, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]:
    ...
@typing.overload
def extract_image_4points(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], corners: list, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]:
    ...
@typing.overload
def extract_image_4points(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], corners: list, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]:
    ...
@typing.overload
def extract_image_4points(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], corners: list, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]:
    ...
@typing.overload
def extract_image_4points(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], corners: list, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
    ...
@typing.overload
def extract_image_4points(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], corners: list, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    ...
@typing.overload
def extract_image_4points(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], corners: list, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    requires 
        - corners is a list of dpoint or line objects. 
        - len(corners) == 4 
        - rows >= 0 
        - columns >= 0 
    ensures 
        - The returned image has the given number of rows and columns. 
        - if (corners contains dpoints) then 
            - The 4 points in corners define a convex quadrilateral and this function 
              extracts that part of the input image img and returns it.  Therefore, 
              each corner of the quadrilateral is associated to a corner of the 
              extracted image and bilinear interpolation and a projective mapping is 
              used to transform the pixels in the quadrilateral into the output image. 
              To determine which corners of the quadrilateral map to which corners of 
              the returned image we fit the tightest possible rectangle to the 
              quadrilateral and map its vertices to their nearest rectangle corners. 
              These corners are then trivially mapped to the output image (i.e.  upper 
              left corner to upper left corner, upper right corner to upper right 
              corner, etc.). 
        - else 
            - This routine finds the 4 intersecting points of the given lines which 
              form a convex quadrilateral and uses them as described above to extract 
              an image.   i.e. It just then calls: extract_image_4points(img, 
              intersections_between_lines, rows, columns). 
            - If no convex quadrilateral can be made from the given lines then this 
              routine throws no_convex_quadrilateral.
    """
@typing.overload
def extract_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], chip_location: chip_details) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def extract_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], chip_location: chip_details) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]:
    ...
@typing.overload
def extract_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], chip_location: chip_details) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]:
    ...
@typing.overload
def extract_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], chip_location: chip_details) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]:
    ...
@typing.overload
def extract_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], chip_location: chip_details) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]:
    ...
@typing.overload
def extract_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], chip_location: chip_details) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]:
    ...
@typing.overload
def extract_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], chip_location: chip_details) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]:
    ...
@typing.overload
def extract_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], chip_location: chip_details) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]:
    ...
@typing.overload
def extract_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], chip_location: chip_details) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
    ...
@typing.overload
def extract_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], chip_location: chip_details) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    ...
@typing.overload
def extract_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], chip_location: chip_details) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
        This routine is just like extract_image_chips() except it takes a single 
        chip_details object and returns a single chip image rather than a list of images.
    """
@typing.overload
def extract_image_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], chip_locations: list) -> list:
    ...
@typing.overload
def extract_image_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], chip_locations: list) -> list:
    ...
@typing.overload
def extract_image_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], chip_locations: list) -> list:
    ...
@typing.overload
def extract_image_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], chip_locations: list) -> list:
    ...
@typing.overload
def extract_image_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], chip_locations: list) -> list:
    ...
@typing.overload
def extract_image_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], chip_locations: list) -> list:
    ...
@typing.overload
def extract_image_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], chip_locations: list) -> list:
    ...
@typing.overload
def extract_image_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], chip_locations: list) -> list:
    ...
@typing.overload
def extract_image_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], chip_locations: list) -> list:
    ...
@typing.overload
def extract_image_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], chip_locations: list) -> list:
    ...
@typing.overload
def extract_image_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], chip_locations: list) -> list:
    """
    requires 
        - for all valid i:  
            - chip_locations[i].rect.is_empty() == false 
            - chip_locations[i].rows*chip_locations[i].cols != 0 
    ensures 
        - This function extracts "chips" from an image.  That is, it takes a list of 
          rectangular sub-windows (i.e. chips) within an image and extracts those 
          sub-windows, storing each into its own image.  It also scales and rotates the 
          image chips according to the instructions inside each chip_details object. 
          It uses bilinear interpolation. 
        - The extracted image chips are returned in a python list of numpy arrays.  The 
          length of the returned array is len(chip_locations). 
        - Let CHIPS be the returned array, then we have: 
            - for all valid i: 
                - #CHIPS[i] == The image chip extracted from the position 
                  chip_locations[i].rect in img. 
                - #CHIPS[i].shape(0) == chip_locations[i].rows 
                - #CHIPS[i].shape(1) == chip_locations[i].cols 
                - The image will have been rotated counter-clockwise by 
                  chip_locations[i].angle radians, around the center of 
                  chip_locations[i].rect, before the chip was extracted.  
        - Any pixels in an image chip that go outside img are set to 0 (i.e. black).
    """
def find_bright_keypoints(xx: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], xy: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], yy: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
    """
    requires 
        - xx, xy, and yy all have the same dimensions. 
    ensures 
        - This routine finds bright "keypoints" in an image.  In general, these are 
          bright/white localized blobs.  It does this by computing the determinant of 
          the image Hessian at each location and storing this value into the returned 
          image if both eigenvalues of the Hessian are negative.  If either eigenvalue 
          is positive then the output value for that pixel is 0.  I.e. 
            - Let OUT denote the returned image. 
            - for all valid r,c: 
                - OUT[r][c] == a number >= 0 and larger values indicate the 
                  presence of a keypoint at this pixel location. 
        - We assume that xx, xy, and yy are the 3 second order gradients of the image 
          in question.  You can obtain these gradients using the image_gradients class. 
        - The output image will have the same dimensions as the input images.
    """
def find_bright_lines(xx: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], xy: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], yy: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> tuple:
    """
    requires 
        - xx, xy, and yy all have the same dimensions. 
    ensures 
        - This routine is similar to sobel_edge_detector(), except instead of finding 
          an edge it finds a bright/white line.  For example, the border between a 
          black piece of paper and a white table is an edge, but a curve drawn with a 
          pencil on a piece of paper makes a line.  Therefore, the output of this 
          routine is a vector field encoded in the horz and vert images, which are 
          returned in a tuple where the first element is horz and the second is vert. 
     
          The vector obtains a large magnitude when centered on a bright line in an image and the 
          direction of the vector is perpendicular to the line.  To be very precise, 
          each vector points in the direction of greatest change in second derivative 
          and the magnitude of the vector encodes the derivative magnitude in that 
          direction.  Moreover, if the second derivative is positive then the output 
          vector is zero.  This zeroing if positive gradients causes the output to be 
          sensitive only to bright lines surrounded by darker pixels. 
     
        - We assume that xx, xy, and yy are the 3 second order gradients of the image 
          in question.  You can obtain these gradients using the image_gradients class. 
        - The output images will have the same dimensions as the input images. 
    """
def find_candidate_object_locations(image: numpy.ndarray, rects: list, kvals: tuple = (50, 200, 3), min_size: typing.SupportsInt | typing.SupportsIndex = 20, max_merging_iterations: typing.SupportsInt | typing.SupportsIndex = 50) -> None:
    """
    Returns found candidate objects
    requires
        - image == an image object which is a numpy ndarray
        - len(kvals) == 3
        - kvals should be a tuple that specifies the range of k values to use.  In
          particular, it should take the form (start, end, num) where num > 0. 
    ensures
        - This function takes an input image and generates a set of candidate
          rectangles which are expected to bound any objects in the image.  It does
          this by running a version of the segment_image() routine on the image and
          then reports rectangles containing each of the segments as well as rectangles
          containing unions of adjacent segments.  The basic idea is described in the
          paper: 
              Segmentation as Selective Search for Object Recognition by Koen E. A. van de Sande, et al.
          Note that this function deviates from what is described in the paper slightly. 
          See the code for details.
        - The basic segmentation is performed kvals[2] times, each time with the k parameter
          (see segment_image() and the Felzenszwalb paper for details on k) set to a different
          value from the range of numbers linearly spaced between kvals[0] to kvals[1].
        - When doing the basic segmentations prior to any box merging, we discard all
          rectangles that have an area < min_size.  Therefore, all outputs and
          subsequent merged rectangles are built out of rectangles that contain at
          least min_size pixels.  Note that setting min_size to a smaller value than
          you might otherwise be interested in using can be useful since it allows a
          larger number of possible merged boxes to be created.
        - There are max_merging_iterations rounds of neighboring blob merging.
          Therefore, this parameter has some effect on the number of output rectangles
          you get, with larger values of the parameter giving more output rectangles.
        - This function appends the output rectangles into #rects.  This means that any
          rectangles in rects before this function was called will still be in there
          after it terminates.  Note further that #rects will not contain any duplicate
          rectangles.  That is, for all valid i and j where i != j it will be true
          that:
            - #rects[i] != rects[j]
    """
def find_dark_keypoints(xx: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], xy: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], yy: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
    """
    requires 
        - xx, xy, and yy all have the same dimensions. 
    ensures 
        - This routine finds dark "keypoints" in an image.  In general, these are 
          dark localized blobs.  It does this by computing the determinant of 
          the image Hessian at each location and storing this value into the returned 
          image if both eigenvalues of the Hessian are negative.  If either eigenvalue 
          is negative then the output value for that pixel is 0.  I.e. 
            - Let OUT denote the returned image. 
            - for all valid r,c: 
                - OUT[r][c] == a number >= 0 and larger values indicate the 
                  presence of a keypoint at this pixel location. 
        - We assume that xx, xy, and yy are the 3 second order gradients of the image 
          in question.  You can obtain these gradients using the image_gradients class. 
        - The output image will have the same dimensions as the input images.
    """
def find_dark_lines(xx: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], xy: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], yy: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> tuple:
    """
    requires 
        - xx, xy, and yy all have the same dimensions. 
    ensures 
        - This routine is similar to sobel_edge_detector(), except instead of finding 
          an edge it finds a dark line.  For example, the border between a black piece 
          of paper and a white table is an edge, but a curve drawn with a pencil on a 
          piece of paper makes a line.  Therefore, the output of this routine is a 
          vector field encoded in the horz and vert images, which are returned in a 
          tuple where the first element is horz and the second is vert. 
     
          The vector obtains a large magnitude when centered on a dark line in an image 
          and the direction of the vector is perpendicular to the line.  To be very 
          precise, each vector points in the direction of greatest change in second 
          derivative and the magnitude of the vector encodes the derivative magnitude 
          in that direction.  Moreover, if the second derivative is negative then the 
          output vector is zero.  This zeroing if negative gradients causes the output 
          to be sensitive only to dark lines surrounded by darker pixels. 
     
        - We assume that xx, xy, and yy are the 3 second order gradients of the image 
          in question.  You can obtain these gradients using the image_gradients class. 
        - The output images will have the same dimensions as the input images. 
    """
def find_line_endpoints(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> points:
    """
    requires 
        - all pixels in img are set to either 255 or 0. 
          (i.e. it must be a binary image) 
    ensures 
        - This routine finds endpoints of lines in a thinned binary image.  For 
          example, if the image was produced by skeleton() or something like a Canny 
          edge detector then you can use find_line_endpoints() to find the pixels 
          sitting on the ends of lines.
    """
@typing.overload
def find_max_global(f: typing.Any, bound1: list, bound2: list, is_integer_variable: list, num_function_calls: typing.SupportsInt | typing.SupportsIndex, solver_epsilon: typing.SupportsFloat | typing.SupportsIndex = 0) -> tuple:
    """
    requires 
        - len(bound1) == len(bound2) == len(is_integer_variable) 
        - for all valid i: bound1[i] != bound2[i] 
        - solver_epsilon >= 0 
        - f() is a real valued multi-variate function.  It must take scalar real 
          numbers as its arguments and the number of arguments must be len(bound1). 
    ensures 
        - This function performs global optimization on the given f() function. 
          The goal is to maximize the following objective function: 
             f(x) 
          subject to the constraints: 
            min(bound1[i],bound2[i]) <= x[i] <= max(bound1[i],bound2[i]) 
            if (is_integer_variable[i]) then x[i] is an integer value (but still 
            represented with float type). 
        - find_max_global() runs until it has called f() num_function_calls times. 
          Then it returns the best x it has found along with the corresponding output 
          of f().  That is, it returns (best_x_seen,f(best_x_seen)).  Here best_x_seen 
          is a list containing the best arguments to f() this function has found. 
        - find_max_global() uses a global optimization method based on a combination of 
          non-parametric global function modeling and quadratic trust region modeling 
          to efficiently find a global maximizer.  It usually does a good job with a 
          relatively small number of calls to f().  For more information on how it 
          works read the documentation for dlib's global_function_search object. 
          However, one notable element is the solver epsilon, which you can adjust. 
     
          The search procedure will only attempt to find a global maximizer to at most 
          solver_epsilon accuracy.  Once a local maximizer is found to that accuracy 
          the search will focus entirely on finding other maxima elsewhere rather than 
          on further improving the current local optima found so far.  That is, once a 
          local maxima is identified to about solver_epsilon accuracy, the algorithm 
          will spend all its time exploring the function to find other local maxima to 
          investigate.  An epsilon of 0 means it will keep solving until it reaches 
          full floating point precision.  Larger values will cause it to switch to pure 
          global exploration sooner and therefore might be more effective if your 
          objective function has many local maxima and you don't care about a super 
          high precision solution. 
        - Any variables that satisfy the following conditions are optimized on a log-scale: 
            - The lower bound on the variable is > 0 
            - The ratio of the upper bound to lower bound is > 1000 
            - The variable is not an integer variable 
          We do this because it's common to optimize machine learning models that have 
          parameters with bounds in a range such as [1e-5 to 1e10] (e.g. the SVM C 
          parameter) and it's much more appropriate to optimize these kinds of 
          variables on a log scale.  So we transform them by applying log() to 
          them and then undo the transform via exp() before invoking the function 
          being optimized.  Therefore, this transformation is invisible to the user 
          supplied functions.  In most cases, it improves the efficiency of the 
          optimizer.
    """
@typing.overload
def find_max_global(f: typing.Any, bound1: list, bound2: list, num_function_calls: typing.SupportsInt | typing.SupportsIndex, solver_epsilon: typing.SupportsFloat | typing.SupportsIndex = 0) -> tuple:
    """
    This function simply calls the other version of find_max_global() with is_integer_variable set to False for all variables.
    """
@typing.overload
def find_min_global(f: typing.Any, bound1: list, bound2: list, is_integer_variable: list, num_function_calls: typing.SupportsInt | typing.SupportsIndex, solver_epsilon: typing.SupportsFloat | typing.SupportsIndex = 0) -> tuple:
    """
    This function is just like find_max_global(), except it performs minimization rather than maximization.
    """
@typing.overload
def find_min_global(f: typing.Any, bound1: list, bound2: list, num_function_calls: typing.SupportsInt | typing.SupportsIndex, solver_epsilon: typing.SupportsFloat | typing.SupportsIndex = 0) -> tuple:
    """
    This function simply calls the other version of find_min_global() with is_integer_variable set to False for all variables.
    """
def find_optimal_momentum_filter(sequence: typing.Any, smoothness: typing.SupportsFloat | typing.SupportsIndex = 1) -> momentum_filter:
    """
    requires
                - sequences.size() != 0
                - for all valid i: sequences[i].size() > 4
                - smoothness >= 0
            ensures
                - This function finds the "optimal" settings of a momentum_filter based on
                  recorded measurement data stored in sequences.  Here we assume that each
                  vector in sequences is a complete track history of some object's measured
                  positions.  What we do is find the momentum_filter that minimizes the
                  following objective function:
                     sum of abs(predicted_location[i] - measured_location[i]) + smoothness*abs(filtered_location[i]-filtered_location[i-1])
                     Where i is a time index.
                  The sum runs over all the data in sequences.  So what we do is find the
                  filter settings that produce smooth filtered trajectories but also produce
                  filtered outputs that are as close to the measured positions as possible.
                  The larger the value of smoothness the less jittery the filter outputs will
                  be, but they might become biased or laggy if smoothness is set really high.
    """
def find_optimal_rect_filter(rects: ..., std: ..., smoothness: typing.SupportsFloat | typing.SupportsIndex = 1) -> rect_filter:
    """
    requires 
        - rects.size() > 4 
        - smoothness >= 0 
    ensures 
        - This function finds the "optimal" settings of a rect_filter based on recorded 
          measurement data stored in rects.  Here we assume that rects is a complete 
          track history of some object's measured positions.  Essentially, what we do 
          is find the rect_filter that minimizes the following objective function: 
             sum of abs(predicted_location[i] - measured_location[i]) + smoothness*abs(filtered_location[i]-filtered_location[i-1]) 
             Where i is a time index. 
          The sum runs over all the data in rects.  So what we do is find the 
          filter settings that produce smooth filtered trajectories but also produce 
          filtered outputs that are as close to the measured positions as possible. 
          The larger the value of smoothness the less jittery the filter outputs will 
          be, but they might become biased or laggy if smoothness is set really high. 
    """
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex, thresh: typing.SupportsFloat | typing.SupportsIndex) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex, thresh: typing.SupportsFloat | typing.SupportsIndex) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex, thresh: typing.SupportsInt | typing.SupportsIndex) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex, thresh: typing.SupportsInt | typing.SupportsIndex) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex, thresh: typing.SupportsInt | typing.SupportsIndex) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex, thresh: typing.SupportsInt | typing.SupportsIndex) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex, thresh: typing.SupportsInt | typing.SupportsIndex) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex, thresh: typing.SupportsInt | typing.SupportsIndex) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex, thresh: typing.SupportsInt | typing.SupportsIndex) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex, thresh: typing.SupportsInt | typing.SupportsIndex) -> points:
    """
    requires 
        - non_max_suppression_radius >= 0 
    ensures 
        - Scans the given image and finds all pixels with values >= thresh that are 
          also local maximums within their 8-connected neighborhood of the image.  Such 
          pixels are collected, sorted in decreasing order of their pixel values, and 
          then non-maximum suppression is applied to this list of points using the 
          given non_max_suppression_radius.  The final list of peaks is then returned. 
     
          Therefore, the returned list, V, will have these properties: 
            - len(V) == the number of peaks found in the image. 
            - When measured in image coordinates, no elements of V are within 
              non_max_suppression_radius distance of each other.  That is, for all valid i!=j 
              it is true that length(V[i]-V[j]) > non_max_suppression_radius. 
            - For each element of V, that element has the maximum pixel value of all 
              pixels in the ball centered on that pixel with radius 
              non_max_suppression_radius.
    """
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex = 0) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex = 0) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex = 0) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex = 0) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex = 0) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex = 0) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex = 0) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex = 0) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex = 0) -> points:
    ...
@typing.overload
def find_peaks(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], non_max_suppression_radius: typing.SupportsFloat | typing.SupportsIndex = 0) -> points:
    """
    performs: return find_peaks(img, non_max_suppression_radius, partition_pixels(img))
    """
@typing.overload
def find_projective_transform(from_points: dpoints, to_points: dpoints) -> point_transform_projective:
    """
    requires 
        - len(from_points) == len(to_points) 
        - len(from_points) >= 4 
    ensures 
        - returns a point_transform_projective object, T, such that for all valid i: 
            length(T(from_points[i]) - to_points[i]) 
          is minimized as often as possible.  That is, this function finds the projective 
          transform that maps points in from_points to points in to_points.  If no 
          projective transform exists which performs this mapping exactly then the one 
          which minimizes the mean squared error is selected. 
    """
@typing.overload
def find_projective_transform(from_points: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], to_points: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> point_transform_projective:
    """
    requires 
        - from_points and to_points have two columns and the same number of rows. 
          Moreover, they have at least 4 rows. 
    ensures 
        - returns a point_transform_projective object, T, such that for all valid i: 
            length(T(dpoint(from_points[i])) - dpoint(to_points[i])) 
          is minimized as often as possible.  That is, this function finds the projective 
          transform that maps points in from_points to points in to_points.  If no 
          projective transform exists which performs this mapping exactly then the one 
          which minimizes the mean squared error is selected. 
    """
@typing.overload
def find_projective_transform(from_points: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], to_points: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> point_transform_projective:
    """
    requires 
        - from_points and to_points have two columns and the same number of rows. 
          Moreover, they have at least 4 rows. 
    ensures 
        - returns a point_transform_projective object, T, such that for all valid i: 
            length(T(dpoint(from_points[i])) - dpoint(to_points[i])) 
          is minimized as often as possible.  That is, this function finds the projective 
          transform that maps points in from_points to points in to_points.  If no 
          projective transform exists which performs this mapping exactly then the one 
          which minimizes the mean squared error is selected. 
    """
@typing.overload
def gaussian_blur(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], sigma: typing.SupportsFloat | typing.SupportsIndex, max_size: typing.SupportsInt | typing.SupportsIndex = 1000) -> tuple:
    ...
@typing.overload
def gaussian_blur(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], sigma: typing.SupportsFloat | typing.SupportsIndex, max_size: typing.SupportsInt | typing.SupportsIndex = 1000) -> tuple:
    ...
@typing.overload
def gaussian_blur(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], sigma: typing.SupportsFloat | typing.SupportsIndex, max_size: typing.SupportsInt | typing.SupportsIndex = 1000) -> tuple:
    ...
@typing.overload
def gaussian_blur(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], sigma: typing.SupportsFloat | typing.SupportsIndex, max_size: typing.SupportsInt | typing.SupportsIndex = 1000) -> tuple:
    ...
@typing.overload
def gaussian_blur(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], sigma: typing.SupportsFloat | typing.SupportsIndex, max_size: typing.SupportsInt | typing.SupportsIndex = 1000) -> tuple:
    ...
@typing.overload
def gaussian_blur(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], sigma: typing.SupportsFloat | typing.SupportsIndex, max_size: typing.SupportsInt | typing.SupportsIndex = 1000) -> tuple:
    """
    requires 
        - sigma > 0 
        - max_size > 0 
        - max_size is an odd number 
    ensures 
        - Filters img with a Gaussian filter of sigma width.  The actual spatial filter will 
          be applied to pixel blocks that are at most max_size wide and max_size tall (note that 
          this function will automatically select a smaller block size as appropriate).  The  
          results are returned.  We also return a rectangle which indicates what pixels 
          in the returned image are considered non-border pixels and therefore contain 
          output from the filter.  E.g. 
            - filtered_img,rect = gaussian_blur(img) 
          would give you the filtered image and the rectangle in question. 
        - The filter is applied to each color channel independently. 
        - Pixels close enough to the edge of img to not have the filter still fit  
          inside the image are set to zero. 
        - The returned image has the same dimensions as the input image.
    """
@typing.overload
def get_face_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], face: full_object_detection, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], face: full_object_detection, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]:
    """
    Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], face: full_object_detection, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]:
    """
    Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], face: full_object_detection, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]:
    """
    Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], face: full_object_detection, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]:
    """
    Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], face: full_object_detection, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]:
    """
    Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], face: full_object_detection, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]:
    """
    Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], face: full_object_detection, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]:
    """
    Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], face: full_object_detection, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
    """
    Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], face: full_object_detection, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """
    Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], face: full_object_detection, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chip_details(det: ..., size: typing.SupportsInt | typing.SupportsIndex = 200, padding: typing.SupportsFloat | typing.SupportsIndex = 0.2) -> chip_details:
    """
    Given a full_object_detection det, returns a chip_details object which can be 
             used to extract an image of given size and padding.
    """
@typing.overload
def get_face_chip_details(dets: ..., std: ..., size: typing.SupportsInt | typing.SupportsIndex = 200, padding: typing.SupportsFloat | typing.SupportsIndex = 0.2) -> chip_detailss:
    """
    Given a list of full_object_detection dets, returns a chip_details object which can be 
             used to extract an image of given size and padding.
    """
@typing.overload
def get_face_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], faces: full_object_detections, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> list:
    """
    Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], faces: full_object_detections, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> list:
    """
    Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], faces: full_object_detections, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> list:
    """
    Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], faces: full_object_detections, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> list:
    """
    Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], faces: full_object_detections, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> list:
    """
    Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], faces: full_object_detections, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> list:
    """
    Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], faces: full_object_detections, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> list:
    """
    Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], faces: full_object_detections, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> list:
    """
    Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], faces: full_object_detections, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> list:
    """
    Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], faces: full_object_detections, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> list:
    """
    Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def get_face_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], faces: full_object_detections, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> list:
    """
    Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
def get_frontal_face_detector() -> ...:
    """
    Returns the default face detector
    """
@typing.overload
def get_histogram(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], hist_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.uint64]:
    ...
@typing.overload
def get_histogram(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], hist_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.uint64]:
    ...
@typing.overload
def get_histogram(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], hist_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.uint64]:
    ...
@typing.overload
def get_histogram(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], hist_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.uint64]:
    """
    ensures 
        - Returns a numpy array, HIST, that contains a histogram of the pixels in img. 
          In particular, we will have: 
            - len(HIST) == hist_size 
            - for all valid i:  
                - HIST[i] == the number of times a pixel with intensity i appears in img.
    """
@typing.overload
def get_rect(img: numpy.ndarray) -> rectangle:
    """
    returns a rectangle(0,0,img.shape(1)-1,img.shape(0)-1).  Therefore, it is the rectangle that bounds the image.
    """
@typing.overload
def get_rect(ht: hough_transform) -> rectangle:
    """
    returns a rectangle(0,0,ht.size()-1,ht.size()-1).  Therefore, it is the rectangle that bounds the Hough transform image.
    """
def grow_rect(rect: rectangle, num: typing.SupportsInt | typing.SupportsIndex) -> rectangle:
    """
    - return shrink_rect(rect, -num) 
      (i.e. grows the given rectangle by expanding its border by num)
    """
def hit_enter_to_continue() -> None:
    """
    Asks the user to hit enter to continue and pauses until they do so.
    """
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], lower_thresh: typing.SupportsInt | typing.SupportsIndex, upper_thresh: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], lower_thresh: typing.SupportsInt | typing.SupportsIndex, upper_thresh: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], lower_thresh: typing.SupportsInt | typing.SupportsIndex, upper_thresh: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], lower_thresh: typing.SupportsInt | typing.SupportsIndex, upper_thresh: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], lower_thresh: typing.SupportsInt | typing.SupportsIndex, upper_thresh: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], lower_thresh: typing.SupportsInt | typing.SupportsIndex, upper_thresh: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], lower_thresh: typing.SupportsInt | typing.SupportsIndex, upper_thresh: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], lower_thresh: typing.SupportsInt | typing.SupportsIndex, upper_thresh: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], lower_thresh: typing.SupportsFloat | typing.SupportsIndex, upper_thresh: typing.SupportsFloat | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], lower_thresh: typing.SupportsFloat | typing.SupportsIndex, upper_thresh: typing.SupportsFloat | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    Applies hysteresis thresholding to img and returns the results.  In particular, 
    pixels in img with values >= upper_thresh have an output value of 255 and all 
    others have a value of 0 unless they are >= lower_thresh and are connected to a 
    pixel with a value >= upper_thresh, in which case they have a value of 255.  Here 
    pixels are connected if there is a path between them composed of pixels that would 
    receive an output of 255.
    """
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def hysteresis_threshold(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    performs: return hysteresis_threshold(img, t1, t2) where the thresholds 
    are first obtained by calling [t1, t2]=partition_pixels(img).
    """
@typing.overload
def insert_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], chip: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], chip_location: chip_details) -> None:
    ...
@typing.overload
def insert_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], chip: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], chip_location: chip_details) -> None:
    ...
@typing.overload
def insert_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], chip: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], chip_location: chip_details) -> None:
    ...
@typing.overload
def insert_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], chip: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], chip_location: chip_details) -> None:
    ...
@typing.overload
def insert_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], chip: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], chip_location: chip_details) -> None:
    ...
@typing.overload
def insert_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], chip: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], chip_location: chip_details) -> None:
    ...
@typing.overload
def insert_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], chip: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], chip_location: chip_details) -> None:
    ...
@typing.overload
def insert_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], chip: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], chip_location: chip_details) -> None:
    ...
@typing.overload
def insert_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], chip: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], chip_location: chip_details) -> None:
    ...
@typing.overload
def insert_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], chip: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], chip_location: chip_details) -> None:
    ...
@typing.overload
def insert_image_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], chip: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], chip_location: chip_details) -> None:
    """
    Inserts chip into img applying the appropriate mapping using chip_locationrequires 
        - img and chip numpy arrays that can be interpreted as images.  They 
          must be the same type of image as well. 
        - the number of rows and columns in chip match the ones in chip_location 
    ensures 
        - This function takes the given chip and inserts it into img using an appropriate 
          mapping, computed from chip_location.
    """
def intersect(a: line, b: line) -> dpoint:
    """
    ensures 
        - returns the point of intersection between lines a and b.  If no such point 
          exists then this function returns a point with Inf values in it.
    """
def inv(trans: point_transform_projective) -> point_transform_projective:
    """
    ensures 
        - If trans is an invertible transformation then this function returns a new 
          transformation that is the inverse of trans. 
    """
@typing.overload
def jet(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def jet(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def jet(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def jet(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def jet(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    Converts a grayscale image into a jet colored image.  This is an image where dark 
    pixels are dark blue and larger values become light blue, then yellow, and then 
    finally red as they approach the maximum pixel values.
    """
def jitter_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], num_jitters: typing.SupportsInt | typing.SupportsIndex = 1, disturb_colors: bool = False) -> list:
    """
    Takes an image and returns a list of jittered images.The returned list contains num_jitters images (default is 1).If disturb_colors is set to True, the colors of the image are disturbed (default is False)
    """
@typing.overload
def label_connected_blobs(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], zero_pixels_are_background: bool = True, neighborhood_connectivity: typing.SupportsInt | typing.SupportsIndex = 8, connected_if_both_not_zero: bool = False) -> tuple:
    ...
@typing.overload
def label_connected_blobs(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], zero_pixels_are_background: bool = True, neighborhood_connectivity: typing.SupportsInt | typing.SupportsIndex = 8, connected_if_both_not_zero: bool = False) -> tuple:
    ...
@typing.overload
def label_connected_blobs(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], zero_pixels_are_background: bool = True, neighborhood_connectivity: typing.SupportsInt | typing.SupportsIndex = 8, connected_if_both_not_zero: bool = False) -> tuple:
    ...
@typing.overload
def label_connected_blobs(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], zero_pixels_are_background: bool = True, neighborhood_connectivity: typing.SupportsInt | typing.SupportsIndex = 8, connected_if_both_not_zero: bool = False) -> tuple:
    ...
@typing.overload
def label_connected_blobs(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], zero_pixels_are_background: bool = True, neighborhood_connectivity: typing.SupportsInt | typing.SupportsIndex = 8, connected_if_both_not_zero: bool = False) -> tuple:
    ...
@typing.overload
def label_connected_blobs(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], zero_pixels_are_background: bool = True, neighborhood_connectivity: typing.SupportsInt | typing.SupportsIndex = 8, connected_if_both_not_zero: bool = False) -> tuple:
    """
    requires 
        - neighborhood_connectivity == 4, 8, or 24 
    ensures 
        - This function labels each of the connected blobs in img with a unique integer  
          label.   
        - An image can be thought of as a graph where pixels A and B are connected if 
          they are close to each other and satisfy some criterion like having the same 
          value or both being non-zero.  Then this function can be understood as 
          labeling all the connected components of this pixel graph such that all 
          pixels in a component get the same label while pixels in different components 
          get different labels.   
        - If zero_pixels_are_background==true then there is a special background component 
          and all pixels with value 0 are assigned to it. Moreover, all such background pixels 
          will always get a blob id of 0 regardless of any other considerations. 
        - This function returns a label image and a count of the number of blobs found. 
          I.e., if you ran this function like: 
            label_img, num_blobs = label_connected_blobs(img) 
          You would obtain the noted label image and number of blobs. 
        - The output label_img has the same dimensions as the input image. 
        - for all valid r and c: 
            - label_img[r][c] == the blob label number for pixel img[r][c].   
            - label_img[r][c] >= 0 
            - if (img[r][c]==0) then 
                - label_img[r][c] == 0 
            - else 
                - label_img[r][c] != 0 
        - if (len(img) != 0) then  
            - The returned num_blobs will be == label_img.max()+1 
              (i.e. returns a number one greater than the maximum blob id number,  
              this is the number of blobs found.) 
        - else 
            - num_blobs will be 0. 
        - blob labels are contiguous, therefore, the number returned by this function is 
          the number of blobs in the image (including the background blob).
    """
@typing.overload
def label_connected_blobs_watershed(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], background_thresh: typing.SupportsInt | typing.SupportsIndex, smoothing: typing.SupportsFloat | typing.SupportsIndex = 0) -> tuple:
    ...
@typing.overload
def label_connected_blobs_watershed(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], background_thresh: typing.SupportsInt | typing.SupportsIndex, smoothing: typing.SupportsFloat | typing.SupportsIndex = 0) -> tuple:
    ...
@typing.overload
def label_connected_blobs_watershed(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], background_thresh: typing.SupportsInt | typing.SupportsIndex, smoothing: typing.SupportsFloat | typing.SupportsIndex = 0) -> tuple:
    ...
@typing.overload
def label_connected_blobs_watershed(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], background_thresh: typing.SupportsFloat | typing.SupportsIndex, smoothing: typing.SupportsFloat | typing.SupportsIndex = 0) -> tuple:
    ...
@typing.overload
def label_connected_blobs_watershed(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], background_thresh: typing.SupportsFloat | typing.SupportsIndex, smoothing: typing.SupportsFloat | typing.SupportsIndex = 0) -> tuple:
    """
    requires 
        - smoothing >= 0 
    ensures 
        - This routine performs a watershed segmentation of the given input image and 
          labels each resulting flooding region with a unique integer label. It does 
          this by marking the brightest pixels as sources of flooding and then flood 
          fills the image outward from those sources.  Each flooded area is labeled 
          with the identity of the source pixel and flooding stops when another flooded 
          area is reached or pixels with values < background_thresh are encountered.   
        - The flooding will also overrun a source pixel if that source pixel has yet to 
          label any neighboring pixels.  This behavior helps to mitigate spurious 
          splits of objects due to noise.  You can further control this behavior by 
          setting the smoothing parameter.  The flooding will take place on an image 
          that has been Gaussian blurred with a sigma==smoothing.  So setting smoothing 
          to a larger number will in general cause more regions to be merged together. 
          Note that the smoothing parameter has no effect on the interpretation of 
          background_thresh since the decision of "background or not background" is 
          always made relative to the unsmoothed input image. 
        - This function returns a tuple of the labeled image and number of blobs found.  
          i.e. you can call it like this: 
            label_img, num_blobs = label_connected_blobs_watershed(img,background_thresh,smoothing) 
        - The returned label_img will have the same dimensions as img.  
        - for all valid r and c: 
            - if (img[r][c] < background_thresh) then 
                - label_img[r][c] == 0, (i.e. the pixel is labeled as background) 
            - else 
                - label_img[r][c] == an integer value indicating the identity of the segment 
                  containing the pixel img[r][c].   
        - The returned num_blobs is the number of labeled segments, including the 
          background segment.  Therefore, the returned number is 1+(the max value in 
          label_img).
    """
@typing.overload
def label_connected_blobs_watershed(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> tuple:
    ...
@typing.overload
def label_connected_blobs_watershed(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> tuple:
    ...
@typing.overload
def label_connected_blobs_watershed(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> tuple:
    ...
@typing.overload
def label_connected_blobs_watershed(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> tuple:
    ...
@typing.overload
def label_connected_blobs_watershed(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> tuple:
    """
    This version of label_connected_blobs_watershed simple invokes: 
       return label_connected_blobs_watershed(img, partition_pixels(img))
    """
@typing.overload
def length(p: point) -> float:
    """
    returns the distance from p to the origin, i.e. the L2 norm of p.
    """
@typing.overload
def length(p: dpoint) -> float:
    """
    returns the distance from p to the origin, i.e. the L2 norm of p.
    """
def load_grayscale_image(filename: str) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    Takes a path and returns a numpy array containing the image, as an 8bit grayscale image.
    """
def load_libsvm_formatted_data(file_name: str) -> tuple:
    """
    ensures    
        - Attempts to read a file of the given name that should contain libsvm    
          formatted data.  The data is returned as a tuple where the first tuple    
          element is an array of sparse vectors and the second element is an array of    
          labels.    
    """
def load_rgb_alpha_image(filename: str) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    Takes a path and returns a numpy array (RGBA) containing the image
    """
def load_rgb_image(filename: str) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    Takes a path and returns a numpy array (RGB) containing the image
    """
def make_bounding_box_regression_training_data(truth: image_dataset_metadata.dataset, detections: typing.Any) -> image_dataset_metadata.dataset:
    """
    requires 
        - len(truth.images) == len(detections) 
        - detections == A dlib.rectangless object or a list of dlib.rectangles. 
    ensures 
        - Suppose you have an object detector that can roughly locate objects in an 
          image.  This means your detector draws boxes around objects, but these are 
          *rough* boxes in the sense that they aren't positioned super accurately.  For 
          instance, HOG based detectors usually have a stride of 8 pixels.  So the 
          positional accuracy is going to be, at best, +/-8 pixels.   
           
          If you want to get better positional accuracy one easy thing to do is train a 
          shape_predictor to give you the corners of the object.  The 
          make_bounding_box_regression_training_data() routine helps you do this by 
          creating an appropriate training dataset.  It does this by taking the dataset 
          you used to train your detector (the truth object), and combining that with 
          the output of your detector on each image in the training dataset (the 
          detections object).  In particular, it will create a new annotated dataset 
          where each object box is one of the rectangles from detections and that 
          object has 4 part annotations, the corners of the truth rectangle 
          corresponding to that detection rectangle.  You can then take the returned 
          dataset and train a shape_predictor on it.  The resulting shape_predictor can 
          then be used to do bounding box regression. 
        - We assume that detections[i] contains object detections corresponding to  
          the image truth.images[i].
    """
@typing.overload
def make_sparse_vector(arg0: sparse_vector) -> None:
    """
    This function modifies its argument so that it is a properly sorted sparse vector.    
    This means that the elements of the sparse vector will be ordered so that pairs    
    with smaller indices come first.  Additionally, there won't be any pairs with    
    identical indices.  If such pairs were present in the input sparse vector then    
    their values will be added together and only one pair with their index will be    
    present in the output.   
    """
@typing.overload
def make_sparse_vector(arg0: sparse_vectors) -> None:
    """
    This function modifies a sparse_vectors object so that all elements it contains are properly sorted sparse vectors.
    """
def max_cost_assignment(cost: matrix) -> list:
    """
    requires    
        - cost.nr() == cost.nc()    
          (i.e. the input must be a square matrix)    
    ensures    
        - Finds and returns the solution to the following optimization problem:    
        
            Maximize: f(A) == assignment_cost(cost, A)    
            Subject to the following constraints:    
                - The elements of A are unique. That is, there aren't any     
                  elements of A which are equal.      
                - len(A) == cost.nr()    
        
        - Note that this function converts the input cost matrix into a 64bit fixed    
          point representation.  Therefore, you should make sure that the values in    
          your cost matrix can be accurately represented by 64bit fixed point values.    
          If this is not the case then the solution my become inaccurate due to    
          rounding error.  In general, this function will work properly when the ratio    
          of the largest to the smallest value in cost is no more than about 1e16.   
    """
def max_index_plus_one(v: sparse_vector) -> int:
    """
    ensures    
        - returns the dimensionality of the given sparse vector.  That is, returns a    
          number one larger than the maximum index value in the vector.  If the vector    
          is empty then returns 0.   
    """
@typing.overload
def max_point(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> dpoint:
    ...
@typing.overload
def max_point(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> dpoint:
    ...
@typing.overload
def max_point(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> dpoint:
    ...
@typing.overload
def max_point(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]) -> dpoint:
    ...
@typing.overload
def max_point(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]) -> dpoint:
    ...
@typing.overload
def max_point(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]) -> dpoint:
    ...
@typing.overload
def max_point(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]) -> dpoint:
    ...
@typing.overload
def max_point(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]) -> dpoint:
    ...
@typing.overload
def max_point(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> dpoint:
    ...
@typing.overload
def max_point(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> dpoint:
    """
    requires 
        - m.size > 0 
    ensures 
        - returns the location of the maximum element of the array, that is, if the 
          returned point is P then it will be the case that: img[P.y,P.x] == img.max().
    """
@typing.overload
def max_point_interpolated(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> dpoint:
    ...
@typing.overload
def max_point_interpolated(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> dpoint:
    ...
@typing.overload
def max_point_interpolated(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> dpoint:
    ...
@typing.overload
def max_point_interpolated(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]) -> dpoint:
    ...
@typing.overload
def max_point_interpolated(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]) -> dpoint:
    ...
@typing.overload
def max_point_interpolated(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]) -> dpoint:
    ...
@typing.overload
def max_point_interpolated(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]) -> dpoint:
    ...
@typing.overload
def max_point_interpolated(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]) -> dpoint:
    ...
@typing.overload
def max_point_interpolated(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> dpoint:
    ...
@typing.overload
def max_point_interpolated(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> dpoint:
    """
    requires 
        - m.size > 0 
    ensures 
        - Like max_point(), this function finds the location in m with the largest 
          value.  However, we additionally use some quadratic interpolation to find the 
          location of the maximum point with sub-pixel accuracy.  Therefore, the 
          returned point is equal to max_point(m) + some small sub-pixel delta.
    """
@typing.overload
def min_barrier_distance(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], iterations: typing.SupportsInt | typing.SupportsIndex = 10, do_left_right_scans: bool = True) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def min_barrier_distance(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], iterations: typing.SupportsInt | typing.SupportsIndex = 10, do_left_right_scans: bool = True) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]:
    ...
@typing.overload
def min_barrier_distance(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], iterations: typing.SupportsInt | typing.SupportsIndex = 10, do_left_right_scans: bool = True) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]:
    ...
@typing.overload
def min_barrier_distance(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], iterations: typing.SupportsInt | typing.SupportsIndex = 10, do_left_right_scans: bool = True) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]:
    ...
@typing.overload
def min_barrier_distance(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], iterations: typing.SupportsInt | typing.SupportsIndex = 10, do_left_right_scans: bool = True) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]:
    ...
@typing.overload
def min_barrier_distance(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], iterations: typing.SupportsInt | typing.SupportsIndex = 10, do_left_right_scans: bool = True) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]:
    ...
@typing.overload
def min_barrier_distance(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], iterations: typing.SupportsInt | typing.SupportsIndex = 10, do_left_right_scans: bool = True) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]:
    ...
@typing.overload
def min_barrier_distance(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], iterations: typing.SupportsInt | typing.SupportsIndex = 10, do_left_right_scans: bool = True) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]:
    ...
@typing.overload
def min_barrier_distance(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], iterations: typing.SupportsInt | typing.SupportsIndex = 10, do_left_right_scans: bool = True) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
    ...
@typing.overload
def min_barrier_distance(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], iterations: typing.SupportsInt | typing.SupportsIndex = 10, do_left_right_scans: bool = True) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    ...
@typing.overload
def min_barrier_distance(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], iterations: typing.SupportsInt | typing.SupportsIndex = 10, do_left_right_scans: bool = True) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    requires 
        - iterations > 0 
    ensures 
        - This function implements the salient object detection method described in the paper: 
            "Minimum barrier salient object detection at 80 fps" by Zhang, Jianming, et al.  
          In particular, we compute the minimum barrier distance between the borders of 
          the image and all the other pixels.  The resulting image is returned.  Note that 
          the paper talks about a bunch of other things you could do beyond computing 
          the minimum barrier distance, but this function doesn't do any of that. It's 
          just the vanilla MBD. 
        - We will perform iterations iterations of MBD passes over the image.  Larger 
          values might give better results but run slower. 
        - During each MBD iteration we make raster scans over the image.  These pass 
          from top->bottom, bottom->top, left->right, and right->left.  If 
          do_left_right_scans==false then the left/right passes are not executed. 
          Skipping them makes the algorithm about 2x faster but might reduce the 
          quality of the output.
    """
@typing.overload
def normalize_image_gradients(img1: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], img2: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> None:
    ...
@typing.overload
def normalize_image_gradients(img1: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], img2: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> None:
    """
    requires 
        - img1 and img2 have the same dimensions. 
    ensures 
        - This function assumes img1 and img2 are the two gradient images produced by a 
          function like sobel_edge_detector().  It then unit normalizes the gradient 
          vectors. That is, for all valid r and c, this function ensures that: 
            - img1[r][c]*img1[r][c] + img2[r][c]*img2[r][c] == 1  
              unless both img1[r][c] and img2[r][c] were 0 initially, then they stay zero.
    """
def num_separable_filters(detector: simple_object_detector) -> int:
    """
    Returns the number of separable filters necessary to represent the HOG filters in the given detector.
    """
@typing.overload
def partition_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> int:
    ...
@typing.overload
def partition_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> int:
    ...
@typing.overload
def partition_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> int:
    ...
@typing.overload
def partition_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> int:
    ...
@typing.overload
def partition_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> float:
    ...
@typing.overload
def partition_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> float:
    """
    Finds a threshold value that would be reasonable to use with 
    threshold_image(img, threshold).  It does this by finding the threshold that 
    partitions the pixels in img into two groups such that the sum of absolute 
    deviations between each pixel and the mean of its group is minimized.
    """
@typing.overload
def partition_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], num_thresholds: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    ...
@typing.overload
def partition_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], num_thresholds: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    ...
@typing.overload
def partition_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], num_thresholds: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    ...
@typing.overload
def partition_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], num_thresholds: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    ...
@typing.overload
def partition_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], num_thresholds: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    ...
@typing.overload
def partition_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], num_thresholds: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    """
    This version of partition_pixels() finds multiple partitions rather than just 
    one partition.  It does this by first partitioning the pixels just as the 
    above partition_pixels(img) does.  Then it forms a new image with only pixels 
    >= that first partition value and recursively partitions this new image. 
    However, the recursion is implemented in an efficient way which is faster than 
    explicitly forming these images and calling partition_pixels(), but the 
    output is the same as if you did.  For example, suppose you called 
    [t1,t2,t2] = partition_pixels(img,3).  Then we would have: 
       - t1 == partition_pixels(img) 
       - t2 == partition_pixels(an image with only pixels with values >= t1 in it) 
       - t3 == partition_pixels(an image with only pixels with values >= t2 in it)
    """
@typing.overload
def polygon_area(pts: dpoints) -> float:
    ...
@typing.overload
def polygon_area(pts: list) -> float:
    """
    ensures 
        - If you walk the points pts in order to make a closed polygon, what is its 
          area?  This function returns that area.  It uses the shoelace formula to 
          compute the result and so works for general non-self-intersecting polygons.
    """
def probability_that_sequence_is_increasing(time_series: typing.Any) -> float:
    """
    returns the probability that the given sequence of real numbers is increasing in value over time.
    """
@typing.overload
def randomly_color_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def randomly_color_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def randomly_color_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    - randomly generates a mapping from gray level pixel values 
      to the RGB pixel space and then uses this mapping to create 
      a colored version of img.  Returns an image which represents 
      this colored version of img. 
    - black pixels in img will remain black in the output image.  
    """
@typing.overload
def reduce(df: _normalized_decision_function_radial_basis, x: vectors, num_basis_vectors: typing.SupportsInt | typing.SupportsIndex, eps: typing.SupportsFloat | typing.SupportsIndex = 0.001) -> _normalized_decision_function_radial_basis:
    ...
@typing.overload
def reduce(df: _normalized_decision_function_radial_basis, x: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], num_basis_vectors: typing.SupportsInt | typing.SupportsIndex, eps: typing.SupportsFloat | typing.SupportsIndex = 0.001) -> _normalized_decision_function_radial_basis:
    """
    requires 
        - eps > 0 
        - num_bv > 0 
    ensures 
        - This routine takes a learned radial basis function and tries to find a 
          new RBF function with num_basis_vectors basis vectors that approximates 
          the given df() as closely as possible.  In particular, it finds a 
          function new_df() such that new_df(x[i])==df(x[i]) as often as possible. 
        - This is accomplished using a reduced set method that begins by using a 
          projection, in kernel space, onto a random set of num_basis_vectors 
          vectors in x.  Then, L-BFGS is used to further optimize new_df() to match 
          df().  The eps parameter controls how long L-BFGS will run, smaller 
          values of eps possibly giving better solutions but taking longer to 
          execute.
    """
def remove_incoherent_edge_pixels(line: points, horz_gradient: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], vert_gradient: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], angle_thresh: typing.SupportsFloat | typing.SupportsIndex) -> points:
    """
    requires 
        - horz_gradient and vert_gradient have the same dimensions. 
        - horz_gradient and vert_gradient represent unit normalized vectors.  That is, 
          you should have called normalize_image_gradients(horz_gradient,vert_gradient) 
          or otherwise caused all the gradients to have unit norm. 
        - for all valid i: 
            get_rect(horz_gradient).contains(line[i]) 
    ensures 
        - This routine looks at all the points in the given line and discards the ones that 
          have outlying gradient directions.  To be specific, this routine returns a set 
          of points PTS such that:  
            - for all valid i,j: 
                - The difference in angle between the gradients for PTS[i] and PTS[j] is  
                  less than angle_threshold degrees.   
            - len(PTS) <= len(line) 
            - PTS is just line with some elements removed.
    """
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    """
    Resizes img, using bilinear interpolation, to have the indicated number of rows and columns.
    """
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    Resizes img, using bilinear interpolation, to have the indicated number of rows and columns.
    """
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], scale: typing.SupportsFloat | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], scale: typing.SupportsFloat | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], scale: typing.SupportsFloat | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], scale: typing.SupportsFloat | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], scale: typing.SupportsFloat | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], scale: typing.SupportsFloat | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    ...
@typing.overload
def resize_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], scale: typing.SupportsFloat | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    Resizes img, using bilinear interpolation, to have the new size (img rows * scale, img cols * scale)
    """
def reverse(l: line) -> line:
    """
    ensures 
        - returns line(l.p2, l.p1) 
          (i.e. returns a line object that represents the same line as l but with the 
          endpoints, and therefore, the normal vector flipped.  This means that the 
          signed distance of operator() is also flipped).
    """
def save_face_chip(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], face: full_object_detection, chip_filename: str, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> None:
    """
    Takes an image and a full_object_detection that references a face in that image and saves the face with the specified file name prefix.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
def save_face_chips(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], faces: full_object_detections, chip_filename: str, size: typing.SupportsInt | typing.SupportsIndex = 150, padding: typing.SupportsFloat | typing.SupportsIndex = 0.25) -> None:
    """
    Takes an image and a full_object_detections object that reference faces in that image and saves the faces with the specified file name prefix.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.
    """
@typing.overload
def save_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], filename: str, quality: typing.SupportsFloat | typing.SupportsIndex = 75) -> None:
    """
    Saves the given image to the specified path. Determines the file type from the file extension specified in the path
    """
@typing.overload
def save_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], filename: str, quality: typing.SupportsFloat | typing.SupportsIndex = 75) -> None:
    """
    Saves the given image to the specified path. Determines the file type from the file extension specified in the path
    """
@typing.overload
def save_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], filename: str, quality: typing.SupportsFloat | typing.SupportsIndex = 75) -> None:
    """
    Saves the given image to the specified path. Determines the file type from the file extension specified in the path
    """
def save_libsvm_formatted_data(file_name: str, samples: sparse_vectors, labels: array) -> None:
    """
    requires    
        - len(samples) == len(labels)    
    ensures    
        - saves the data to the given file in libsvm format   
    """
def scale_rect(rect: rectangle, scale: typing.SupportsFloat | typing.SupportsIndex) -> rectangle:
    """
    - return scale_rect(rect, scale) 
    (i.e. resizes the given rectangle by a scale factor)
    """
def set_dnn_prefer_smallest_algorithms() -> None:
    """
    Tells cuDNN to use slower algorithms that use less RAM.
    """
def shrink_rect(rect: rectangle, num: typing.SupportsInt | typing.SupportsIndex) -> rectangle:
    """
     returns rectangle(rect.left()+num, rect.top()+num, rect.right()-num, rect.bottom()-num) 
      (i.e. shrinks the given rectangle by shrinking its border by num)
    """
@typing.overload
def signed_distance_to_line(l: line, p: point) -> float:
    ...
@typing.overload
def signed_distance_to_line(l: line, p: dpoint) -> float:
    """
    ensures 
        - returns how far p is from the line l.  This is a signed distance.  The sign 
          indicates which side of the line the point is on and the magnitude is the 
          distance. Moreover, the direction of positive sign is pointed to by the 
          vector l.normal. 
        - To be specific, this routine returns dot(p-l.p1, l.normal)
    """
def skeleton(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    requires 
        - all pixels in img are set to either 255 or 0. 
    ensures 
        - This function computes the skeletonization of img and stores the result in 
          #img.  That is, given a binary image, we progressively thin the binary blobs 
          (composed of on_pixel values) until only a single pixel wide skeleton of the 
          original blobs remains. 
        - Doesn't change the shape or size of img.
    """
@typing.overload
def sobel_edge_detector(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> tuple:
    ...
@typing.overload
def sobel_edge_detector(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> tuple:
    ...
@typing.overload
def sobel_edge_detector(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> tuple:
    ...
@typing.overload
def sobel_edge_detector(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]) -> tuple:
    ...
@typing.overload
def sobel_edge_detector(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]) -> tuple:
    ...
@typing.overload
def sobel_edge_detector(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]) -> tuple:
    ...
@typing.overload
def sobel_edge_detector(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]) -> tuple:
    ...
@typing.overload
def sobel_edge_detector(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]) -> tuple:
    ...
@typing.overload
def sobel_edge_detector(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> tuple:
    ...
@typing.overload
def sobel_edge_detector(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> tuple:
    """
    Applies the sobel edge detector to the given input image and returns two gradient 
    images in a tuple.  The first contains the x gradients and the second contains the 
    y gradients of the image.
    """
def solve_structural_svm_problem(problem: typing.Any) -> vector:
    """
    This function solves a structural SVM problem and returns the weight vector    
    that defines the solution.  See the example program python_examples/svm_struct.py    
    for documentation about how to create a proper problem object.   
    """
@typing.overload
def spatially_filter_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], filter: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> tuple:
    ...
@typing.overload
def spatially_filter_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], filter: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> tuple:
    ...
@typing.overload
def spatially_filter_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], filter: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> tuple:
    """
    requires 
        - filter.size != 0 
    ensures 
        - Applies the given spatial filter to img and returns the result (i.e. we  
          cross-correlate img with filter).  We also return a rectangle which 
          indicates what pixels in the returned image are considered non-border pixels 
          and therefore contain output from the filter.  E.g. 
            - filtered_img,rect = spatially_filter_image(img, filter) 
          would give you the filtered image and the rectangle in question.  Since the 
          returned image has the same shape as img we fill the border pixels by setting 
          them to 0. 
     
        - The filter is applied such that it's centered over the pixel it writes its 
          output into.  For centering purposes, we consider the center element of the 
          filter to be filter[filter.shape[0]/2,filter.shape[1]/2].  This means that 
          the filter that writes its output to a pixel at location point(c,r) and is W 
          by H (width by height) pixels in size operates on exactly the pixels in the 
          rectangle centered_rect(point(c,r),W,H) within img.
    """
@typing.overload
def spatially_filter_image_separable(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], row_filter: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], col_filter: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8]) -> tuple:
    ...
@typing.overload
def spatially_filter_image_separable(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], row_filter: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], col_filter: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> tuple:
    ...
@typing.overload
def spatially_filter_image_separable(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], row_filter: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], col_filter: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> tuple:
    """
    requires 
        - row_filter.size != 0 
        - col_filter.size != 0 
        - row_filter and col_filter are both either row or column vectors.  
    ensures 
        - Applies the given separable spatial filter to img and returns the result 
          (i.e. we cross-correlate img with the filters).  In particular, calling this 
          function has the same effect as calling the regular spatially_filter_image() 
          routine with a filter, FILT, defined as follows:  
            - FILT(r,c) == col_filter(r)*row_filter(c) 
          Therefore, the return value of this routine is the same as if it were 
          implemented as:    
            return spatially_filter_image(img, FILT) 
          Except that this version should be faster for separable filters.
    """
@typing.overload
def sub_image(img: numpy.ndarray, rect: rectangle) -> numpy.ndarray:
    """
    Returns a new numpy array that references the sub window in img defined by rect. 
    If rect is larger than img then rect is cropped so that it does not go outside img. 
    Therefore, this routine is equivalent to performing: 
        win = get_rect(img).intersect(rect) 
        subimg = img[win.top():win.bottom()-1,win.left():win.right()-1]
    """
@typing.overload
def sub_image(image_and_rect_tuple: tuple) -> numpy.ndarray:
    """
    Performs: return sub_image(image_and_rect_tuple[0], image_and_rect_tuple[1])
    """
@typing.overload
def suppress_non_maximum_edges(horz: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], vert: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
    """
    requires 
        - The two input images have the same dimensions. 
    ensures 
        - Returns an image, of the same dimensions as the input.  Each element in this 
          image holds the edge strength at that location.  Moreover, edge pixels that are not  
          local maximizers have been set to 0. 
        - let edge_strength(r,c) == sqrt(pow(horz[r][c],2) + pow(vert[r][c],2)) 
          (i.e. The Euclidean norm of the gradient) 
        - let OUT denote the returned image. 
        - for all valid r and c: 
            - if (edge_strength(r,c) is at a maximum with respect to its 2 neighboring 
              pixels along the line indicated by the image gradient vector (horz[r][c],vert[r][c])) then 
                - OUT[r][c] == edge_strength(r,c) 
            - else 
                - OUT[r][c] == 0
    """
@typing.overload
def suppress_non_maximum_edges(horz_and_vert_gradients: tuple) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
    """
    Performs: return suppress_non_maximum_edges(horz_and_vert_gradients[0], horz_and_vert_gradients[1])
    """
@typing.overload
def test_binary_decision_function(function: _normalized_decision_function_radial_basis, samples: vectors, labels: array) -> binary_test:
    ...
@typing.overload
def test_binary_decision_function(function: _normalized_decision_function_radial_basis, samples: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], labels: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> binary_test:
    ...
@typing.overload
def test_binary_decision_function(function: _decision_function_linear, samples: vectors, labels: array) -> binary_test:
    ...
@typing.overload
def test_binary_decision_function(function: _decision_function_sparse_linear, samples: sparse_vectors, labels: array) -> binary_test:
    ...
@typing.overload
def test_binary_decision_function(function: _decision_function_radial_basis, samples: vectors, labels: array) -> binary_test:
    ...
@typing.overload
def test_binary_decision_function(function: _decision_function_sparse_radial_basis, samples: sparse_vectors, labels: array) -> binary_test:
    ...
@typing.overload
def test_binary_decision_function(function: _decision_function_polynomial, samples: vectors, labels: array) -> binary_test:
    ...
@typing.overload
def test_binary_decision_function(function: _decision_function_sparse_polynomial, samples: sparse_vectors, labels: array) -> binary_test:
    ...
@typing.overload
def test_binary_decision_function(function: _decision_function_histogram_intersection, samples: vectors, labels: array) -> binary_test:
    ...
@typing.overload
def test_binary_decision_function(function: _decision_function_sparse_histogram_intersection, samples: sparse_vectors, labels: array) -> binary_test:
    ...
@typing.overload
def test_binary_decision_function(function: _decision_function_sigmoid, samples: vectors, labels: array) -> binary_test:
    ...
@typing.overload
def test_binary_decision_function(function: _decision_function_sparse_sigmoid, samples: sparse_vectors, labels: array) -> binary_test:
    ...
@typing.overload
def test_ranking_function(function: _decision_function_linear, samples: ranking_pairs) -> ranking_test:
    ...
@typing.overload
def test_ranking_function(function: _decision_function_sparse_linear, samples: sparse_ranking_pairs) -> ranking_test:
    ...
@typing.overload
def test_ranking_function(function: _decision_function_linear, sample: ranking_pair) -> ranking_test:
    ...
@typing.overload
def test_ranking_function(function: _decision_function_sparse_linear, sample: sparse_ranking_pair) -> ranking_test:
    ...
@typing.overload
def test_regression_function(function: _decision_function_linear, samples: vectors, targets: array) -> regression_test:
    ...
@typing.overload
def test_regression_function(function: _decision_function_sparse_linear, samples: sparse_vectors, targets: array) -> regression_test:
    ...
@typing.overload
def test_regression_function(function: _decision_function_radial_basis, samples: vectors, targets: array) -> regression_test:
    ...
@typing.overload
def test_regression_function(function: _decision_function_sparse_radial_basis, samples: sparse_vectors, targets: array) -> regression_test:
    ...
@typing.overload
def test_regression_function(function: _decision_function_histogram_intersection, samples: vectors, targets: array) -> regression_test:
    ...
@typing.overload
def test_regression_function(function: _decision_function_sparse_histogram_intersection, samples: sparse_vectors, targets: array) -> regression_test:
    ...
@typing.overload
def test_regression_function(function: _decision_function_sigmoid, samples: vectors, targets: array) -> regression_test:
    ...
@typing.overload
def test_regression_function(function: _decision_function_sparse_sigmoid, samples: sparse_vectors, targets: array) -> regression_test:
    ...
@typing.overload
def test_regression_function(function: _decision_function_polynomial, samples: vectors, targets: array) -> regression_test:
    ...
@typing.overload
def test_regression_function(function: _decision_function_sparse_polynomial, samples: sparse_vectors, targets: array) -> regression_test:
    ...
@typing.overload
def test_sequence_segmenter(arg0: segmenter_type, arg1: vectorss, arg2: rangess) -> segmenter_test:
    ...
@typing.overload
def test_sequence_segmenter(arg0: segmenter_type, arg1: sparse_vectorss, arg2: rangess) -> segmenter_test:
    ...
@typing.overload
def test_shape_predictor(dataset_filename: str, predictor_filename: str) -> float:
    """
    ensures 
        - Loads an image dataset from dataset_filename.  We assume dataset_filename is 
          a file using the XML format written by save_image_dataset_metadata(). 
        - Loads a shape_predictor from the file predictor_filename.  This means 
          predictor_filename should be a file produced by the train_shape_predictor() 
          routine. 
        - This function tests the predictor against the dataset and returns the 
          mean average error of the detector.  In fact, The 
          return value of this function is identical to that of dlib's 
          shape_predictor_trainer() routine.  Therefore, see the documentation 
          for shape_predictor_trainer() for a detailed definition of the mean average error.
    """
@typing.overload
def test_shape_predictor(images: list, detections: list, shape_predictor: shape_predictor) -> float:
    """
    requires 
        - len(images) == len(object_detections) 
        - images should be a list of numpy matrices that represent images, either RGB or grayscale. 
        - object_detections should be a list of lists of dlib.full_object_detection objects.       Each dlib.full_object_detection contains the bounding box and the lists of points that make up the object parts.
     ensures 
        - shape_predictor should be a file produced by the train_shape_predictor()  
          routine. 
        - This function tests the predictor against the dataset and returns the 
          mean average error of the detector.  In fact, The 
          return value of this function is identical to that of dlib's 
          shape_predictor_trainer() routine.  Therefore, see the documentation 
          for shape_predictor_trainer() for a detailed definition of the mean average error.
    """
@typing.overload
def test_shape_predictor(images: list, detections: list, scales: list, shape_predictor: shape_predictor) -> float:
    """
    requires 
        - len(images) == len(object_detections) 
        - len(object_detections) == len(scales) 
        - for every sublist in object_detections: len(object_detections[i]) == len(scales[i]) 
        - scales is a list of floating point scales that each predicted part location       should be divided by. Useful for normalization. 
        - images should be a list of numpy matrices that represent images, either RGB or grayscale. 
        - object_detections should be a list of lists of dlib.full_object_detection objects.       Each dlib.full_object_detection contains the bounding box and the lists of points that make up the object parts.
     ensures 
        - shape_predictor should be a file produced by the train_shape_predictor()  
          routine. 
        - This function tests the predictor against the dataset and returns the 
          mean average error of the detector.  In fact, The 
          return value of this function is identical to that of dlib's 
          shape_predictor_trainer() routine.  Therefore, see the documentation 
          for shape_predictor_trainer() for a detailed definition of the mean average error.
    """
@typing.overload
def test_simple_object_detector(dataset_filename: str, detector_filename: str, upsampling_amount: typing.SupportsInt | typing.SupportsIndex = -1) -> simple_test_results:
    """
    ensures 
                    - Loads an image dataset from dataset_filename.  We assume dataset_filename is 
                      a file using the XML format written by save_image_dataset_metadata(). 
                    - Loads a simple_object_detector from the file detector_filename.  This means 
                      detector_filename should be a file produced by the train_simple_object_detector()  
                      routine. 
                    - This function tests the detector against the dataset and returns the 
                      precision, recall, and average precision of the detector.  In fact, The 
                      return value of this function is identical to that of dlib's 
                      test_object_detection_function() routine.  Therefore, see the documentation 
                      for test_object_detection_function() for a detailed definition of these 
                      metrics. 
                    - if upsampling_amount>=0 then we upsample the data by upsampling_amount rather than 
                      use any upsampling amount that happens to be encoded in the given detector.  If upsampling_amount<0 
                      then we use the upsampling amount the detector wants to use.
    """
@typing.overload
def test_simple_object_detector(dataset_filename: str, detector: ..., upsampling_amount: typing.SupportsInt | typing.SupportsIndex = -1) -> simple_test_results:
    """
    ensures 
                    - Loads an image dataset from dataset_filename.  We assume dataset_filename is 
                      a file using the XML format written by save_image_dataset_metadata(). 
                    - Loads a simple_object_detector from the file detector_filename.  This means 
                      detector_filename should be a file produced by the train_simple_object_detector()  
                      routine. 
                    - This function tests the detector against the dataset and returns the 
                      precision, recall, and average precision of the detector.  In fact, The 
                      return value of this function is identical to that of dlib's 
                      test_object_detection_function() routine.  Therefore, see the documentation 
                      for test_object_detection_function() for a detailed definition of these 
                      metrics. 
                    - if upsampling_amount>=0 then we upsample the data by upsampling_amount rather than 
                      use any upsampling amount that happens to be encoded in the given detector.  If upsampling_amount<0 
                      then we use the upsampling amount the detector wants to use.
    """
@typing.overload
def test_simple_object_detector(images: list, boxes: list, detector: ..., dlib: ..., upsampling_amount: typing.SupportsInt | typing.SupportsIndex = 0) -> simple_test_results:
    """
    requires 
                   - len(images) == len(boxes) 
                   - images should be a list of numpy matrices that represent images, either RGB or grayscale. 
                   - boxes should be a list of lists of dlib.rectangle object. 
                   - Optionally, take the number of times to upsample the testing images (upsampling_amount >= 0). 
                 ensures 
                   - Loads a simple_object_detector from the file detector_filename.  This means 
                     detector_filename should be a file produced by the train_simple_object_detector() 
                     routine. 
                   - This function tests the detector against the dataset and returns the 
                     precision, recall, and average precision of the detector.  In fact, The 
                     return value of this function is identical to that of dlib's 
                     test_object_detection_function() routine.  Therefore, see the documentation 
                     for test_object_detection_function() for a detailed definition of these 
                     metrics. 
    """
@typing.overload
def test_simple_object_detector(images: list, boxes: list, detector: ..., upsampling_amount: typing.SupportsInt | typing.SupportsIndex = -1) -> simple_test_results:
    """
    requires 
                   - len(images) == len(boxes) 
                   - images should be a list of numpy matrices that represent images, either RGB or grayscale. 
                   - boxes should be a list of lists of dlib.rectangle object. 
                 ensures 
                   - Loads a simple_object_detector from the file detector_filename.  This means 
                     detector_filename should be a file produced by the train_simple_object_detector() 
                     routine. 
                   - This function tests the detector against the dataset and returns the 
                     precision, recall, and average precision of the detector.  In fact, The 
                     return value of this function is identical to that of dlib's 
                     test_object_detection_function() routine.  Therefore, see the documentation 
                     for test_object_detection_function() for a detailed definition of these 
                     metrics. 
    """
def threshold_filter_singular_values(detector: simple_object_detector, thresh: typing.SupportsFloat | typing.SupportsIndex) -> simple_object_detector:
    """
    requires 
        - thresh >= 0 
    ensures 
        - Removes all components of the filters in the given detector that have 
          singular values that are smaller than the given threshold.  Therefore, this 
          function allows you to control how many separable filters are in a detector. 
          In particular, as thresh gets larger the quantity 
          num_separable_filters(threshold_filter_singular_values(detector,thresh)) 
          will generally get smaller and therefore give a faster running detector. 
          However, note that at some point a large enough thresh will drop too much 
          information from the filters and their accuracy will suffer.   
        - returns the updated detector
    """
@typing.overload
def threshold_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def threshold_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def threshold_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def threshold_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def threshold_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def threshold_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    Thresholds img and returns the result.  Pixels in img with grayscale values >= partition_pixels(img) 
    have an output value of 255 and all others have a value of 0.
    """
@typing.overload
def threshold_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], thresh: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def threshold_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], thresh: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def threshold_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], thresh: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def threshold_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], thresh: typing.SupportsFloat | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def threshold_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], thresh: typing.SupportsFloat | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def threshold_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], thresh: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    Thresholds img and returns the result.  Pixels in img with grayscale values >= thresh 
    have an output value of 255 and all others have a value of 0.
    """
def tile_images(images: list) -> numpy.ndarray:
    """
    requires 
        - images is a list of numpy arrays that can be interpreted as images.  They 
          must all be the same type of image as well. 
    ensures 
        - This function takes the given images and tiles them into a single large 
          square image and returns this new big tiled image.  Therefore, it is a 
          useful method to visualize many small images at once.
    """
@typing.overload
def train_sequence_segmenter(*args, **kwargs) -> segmenter_type:
    ...
@typing.overload
def train_sequence_segmenter(*args, **kwargs) -> segmenter_type:
    ...
@typing.overload
def train_shape_predictor(images: list, object_detections: list, options: shape_predictor_training_options) -> shape_predictor:
    """
    requires 
        - options.lambda_param > 0 
        - 0 < options.nu <= 1 
        - options.feature_pool_region_padding >= 0 
        - len(images) == len(object_detections) 
        - images should be a list of numpy matrices that represent images, either RGB or grayscale. 
        - object_detections should be a list of lists of dlib.full_object_detection objects.       Each dlib.full_object_detection contains the bounding box and the lists of points that make up the object parts.
    ensures 
        - Uses dlib's shape_predictor_trainer object to train a 
          shape_predictor based on the provided labeled images, full_object_detections, and options.
        - The trained shape_predictor is returned
    """
@typing.overload
def train_shape_predictor(dataset_filename: str, predictor_output_filename: str, options: shape_predictor_training_options) -> None:
    """
    requires 
        - options.lambda_param > 0 
        - 0 < options.nu <= 1 
        - options.feature_pool_region_padding >= 0 
    ensures 
        - Uses dlib's shape_predictor_trainer to train a 
          shape_predictor based on the labeled images in the XML file 
          dataset_filename and the provided options.  This function assumes the file dataset_filename is in the 
          XML format produced by dlib's save_image_dataset_metadata() routine. 
        - The trained shape predictor is serialized to the file predictor_output_filename.
    """
@typing.overload
def train_simple_object_detector(dataset_filename: str, detector_output_filename: str, options: simple_object_detector_training_options) -> None:
    """
    requires 
        - options.C > 0 
    ensures 
        - Uses the structural_object_detection_trainer to train a 
          simple_object_detector based on the labeled images in the XML file 
          dataset_filename.  This function assumes the file dataset_filename is in the 
          XML format produced by dlib's save_image_dataset_metadata() routine. 
        - This function will apply a reasonable set of default parameters and 
          preprocessing techniques to the training procedure for simple_object_detector 
          objects.  So the point of this function is to provide you with a very easy 
          way to train a basic object detector.   
        - The trained object detector is serialized to the file detector_output_filename.
    """
@typing.overload
def train_simple_object_detector(images: list, boxes: list, options: simple_object_detector_training_options) -> ...:
    """
    requires 
        - options.C > 0 
        - len(images) == len(boxes) 
        - images should be a list of numpy matrices that represent images, either RGB or grayscale. 
        - boxes should be a list of lists of dlib.rectangle object. 
    ensures 
        - Uses the structural_object_detection_trainer to train a 
          simple_object_detector based on the labeled images and bounding boxes.  
        - This function will apply a reasonable set of default parameters and 
          preprocessing techniques to the training procedure for simple_object_detector 
          objects.  So the point of this function is to provide you with a very easy 
          way to train a basic object detector.   
        - The trained object detector is returned.
    """
@typing.overload
def transform_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], map_point: point_transform_projective, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    ...
@typing.overload
def transform_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], map_point: point_transform_projective, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]]:
    ...
@typing.overload
def transform_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], map_point: point_transform_projective, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]]:
    ...
@typing.overload
def transform_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], map_point: point_transform_projective, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]]:
    ...
@typing.overload
def transform_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], map_point: point_transform_projective, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]]:
    ...
@typing.overload
def transform_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], map_point: point_transform_projective, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]]:
    ...
@typing.overload
def transform_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], map_point: point_transform_projective, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]]:
    ...
@typing.overload
def transform_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], map_point: point_transform_projective, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]]:
    ...
@typing.overload
def transform_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], map_point: point_transform_projective, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
    ...
@typing.overload
def transform_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], map_point: point_transform_projective, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]:
    ...
@typing.overload
def transform_image(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], map_point: point_transform_projective, rows: typing.SupportsInt | typing.SupportsIndex, columns: typing.SupportsInt | typing.SupportsIndex) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    """
    requires 
        - rows > 0 
        - columns > 0 
    ensures 
        - Returns an image that is the given rows by columns in size and contains a 
          transformed part of img.  To do this, we interpret map_point as a mapping 
          from pixels in the returned image to pixels in the input img.  transform_image()  
          uses this mapping and bilinear interpolation to fill the output image with an 
          interpolated copy of img.   
        - Any locations in the output image that map to pixels outside img are set to 0.
    """
@typing.overload
def translate_rect(rect: rectangle, p: point) -> rectangle:
    """
     returns rectangle(rect.left()+p.x, rect.top()+p.y, rect.right()+p.x, rect.bottom()+p.y) 
      (i.e. moves the location of the rectangle but doesn't change its shape)
    """
@typing.overload
def translate_rect(rect: drectangle, p: point) -> drectangle:
    """
     returns rectangle(rect.left()+p.x, rect.top()+p.y, rect.right()+p.x, rect.bottom()+p.y) 
      (i.e. moves the location of the rectangle but doesn't change its shape)
    """
@typing.overload
def translate_rect(rect: rectangle, p: dpoint) -> rectangle:
    """
     returns rectangle(rect.left()+p.x, rect.top()+p.y, rect.right()+p.x, rect.bottom()+p.y) 
      (i.e. moves the location of the rectangle but doesn't change its shape)
    """
@typing.overload
def translate_rect(rect: drectangle, p: dpoint) -> drectangle:
    """
     returns rectangle(rect.left()+p.x, rect.top()+p.y, rect.right()+p.x, rect.bottom()+p.y) 
      (i.e. moves the location of the rectangle but doesn't change its shape)
    """
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], x_border_size: typing.SupportsInt | typing.SupportsIndex, y_border_size: typing.SupportsInt | typing.SupportsIndex) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], x_border_size: typing.SupportsInt | typing.SupportsIndex, y_border_size: typing.SupportsInt | typing.SupportsIndex) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], x_border_size: typing.SupportsInt | typing.SupportsIndex, y_border_size: typing.SupportsInt | typing.SupportsIndex) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], x_border_size: typing.SupportsInt | typing.SupportsIndex, y_border_size: typing.SupportsInt | typing.SupportsIndex) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], x_border_size: typing.SupportsInt | typing.SupportsIndex, y_border_size: typing.SupportsInt | typing.SupportsIndex) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], x_border_size: typing.SupportsInt | typing.SupportsIndex, y_border_size: typing.SupportsInt | typing.SupportsIndex) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], x_border_size: typing.SupportsInt | typing.SupportsIndex, y_border_size: typing.SupportsInt | typing.SupportsIndex) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], x_border_size: typing.SupportsInt | typing.SupportsIndex, y_border_size: typing.SupportsInt | typing.SupportsIndex) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], x_border_size: typing.SupportsInt | typing.SupportsIndex, y_border_size: typing.SupportsInt | typing.SupportsIndex) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], x_border_size: typing.SupportsInt | typing.SupportsIndex, y_border_size: typing.SupportsInt | typing.SupportsIndex) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], x_border_size: typing.SupportsInt | typing.SupportsIndex, y_border_size: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    requires 
        - x_border_size >= 0 
        - y_border_size >= 0 
    ensures 
        - The size and shape of img isn't changed by this function. 
        - for all valid r such that r+y_border_size or r-y_border_size gives an invalid row 
            - for all valid c such that c+x_border_size or c-x_border_size gives an invalid column  
                - assigns the pixel img[r][c] to 0.  
                  (i.e. assigns 0 to every pixel in the border of img)
    """
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], inside: rectangle) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint16]], inside: rectangle) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint32]], inside: rectangle) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]], inside: rectangle) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int8]], inside: rectangle) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int16]], inside: rectangle) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int32]], inside: rectangle) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.int64]], inside: rectangle) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]], inside: rectangle) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]], inside: rectangle) -> None:
    ...
@typing.overload
def zero_border_pixels(img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]], inside: rectangle) -> None:
    """
    ensures 
        - The size and shape of img isn't changed by this function. 
        - All the pixels in img that are not contained inside the inside rectangle 
          given to this function are set to 0.  That is, anything not "inside" is on 
          the border and set to 0.
    """
DLIB_USE_BLAS: bool = True
DLIB_USE_CUDA: bool = False
DLIB_USE_LAPACK: bool = True
KBD_MOD_ALT: keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_ALT: 4>
KBD_MOD_CAPS_LOCK: keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_CAPS_LOCK: 16>
KBD_MOD_CONTROL: keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_CONTROL: 2>
KBD_MOD_META: keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_META: 8>
KBD_MOD_NONE: keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_NONE: 0>
KBD_MOD_NUM_LOCK: keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_NUM_LOCK: 32>
KBD_MOD_SCROLL_LOCK: keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_SCROLL_LOCK: 64>
KBD_MOD_SHIFT: keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_SHIFT: 1>
KEY_ALT: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_ALT: 3>
KEY_BACKSPACE: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_BACKSPACE: 0>
KEY_CAPS_LOCK: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_CAPS_LOCK: 5>
KEY_CTRL: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_CTRL: 2>
KEY_DELETE: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_DELETE: 16>
KEY_DOWN: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_DOWN: 14>
KEY_END: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_END: 9>
KEY_ESC: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_ESC: 6>
KEY_F1: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F1: 18>
KEY_F10: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F10: 27>
KEY_F11: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F11: 28>
KEY_F12: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F12: 29>
KEY_F2: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F2: 19>
KEY_F3: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F3: 20>
KEY_F4: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F4: 21>
KEY_F5: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F5: 22>
KEY_F6: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F6: 23>
KEY_F7: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F7: 24>
KEY_F8: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F8: 25>
KEY_F9: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F9: 26>
KEY_HOME: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_HOME: 10>
KEY_INSERT: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_INSERT: 15>
KEY_LEFT: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_LEFT: 11>
KEY_PAGE_DOWN: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_PAGE_DOWN: 8>
KEY_PAGE_UP: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_PAGE_UP: 7>
KEY_PAUSE: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_PAUSE: 4>
KEY_RIGHT: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_RIGHT: 12>
KEY_SCROLL_LOCK: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_SCROLL_LOCK: 17>
KEY_SHIFT: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_SHIFT: 1>
KEY_UP: non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_UP: 13>
USE_AVX_INSTRUCTIONS: bool = False
USE_NEON_INSTRUCTIONS: bool = True
__time_compiled__: str = 'Aug  4 2026 11:41:15'
__version__: str = '20.0.1'
