from __future__ import annotations
import _dlib_pybind11
from _dlib_pybind11 import angle_between_lines
from _dlib_pybind11 import apply_cca_transform
from _dlib_pybind11 import array
from _dlib_pybind11 import as_grayscale
from _dlib_pybind11 import assignment_cost
from _dlib_pybind11 import auto_train_rbf_classifier
from _dlib_pybind11 import bottom_up_clustering
from _dlib_pybind11 import cca
from _dlib_pybind11 import cca_outputs
from _dlib_pybind11 import center
from _dlib_pybind11 import centered_rect
from _dlib_pybind11 import centered_rects
from _dlib_pybind11 import chinese_whispers
from _dlib_pybind11 import chinese_whispers_clustering
from _dlib_pybind11 import chip_details
from _dlib_pybind11 import chip_detailss
from _dlib_pybind11 import chip_dims
from _dlib_pybind11 import cnn_face_detection_model_v1
from _dlib_pybind11 import convert_image
from _dlib_pybind11 import convert_image_scaled
from _dlib_pybind11 import correlation_tracker
from _dlib_pybind11 import count_points_between_lines
from _dlib_pybind11 import count_points_on_side_of_line
from _dlib_pybind11 import count_steps_without_decrease
from _dlib_pybind11 import count_steps_without_decrease_robust
from _dlib_pybind11 import cross_validate_ranking_trainer
from _dlib_pybind11 import cross_validate_sequence_segmenter
from _dlib_pybind11 import cross_validate_trainer
from _dlib_pybind11 import cross_validate_trainer_threaded
from _dlib_pybind11 import cuda
from _dlib_pybind11 import distance_to_line
from _dlib_pybind11 import dot
from _dlib_pybind11 import dpoint
from _dlib_pybind11 import dpoints
from _dlib_pybind11 import drectangle
from _dlib_pybind11 import equalize_histogram
from _dlib_pybind11 import extract_image_4points
from _dlib_pybind11 import extract_image_chip
from _dlib_pybind11 import extract_image_chips
from _dlib_pybind11 import face_recognition_model_v1
from _dlib_pybind11 import fhog_object_detector
from _dlib_pybind11 import find_bright_keypoints
from _dlib_pybind11 import find_bright_lines
from _dlib_pybind11 import find_candidate_object_locations
from _dlib_pybind11 import find_dark_keypoints
from _dlib_pybind11 import find_dark_lines
from _dlib_pybind11 import find_line_endpoints
from _dlib_pybind11 import find_max_global
from _dlib_pybind11 import find_min_global
from _dlib_pybind11 import find_optimal_momentum_filter
from _dlib_pybind11 import find_optimal_rect_filter
from _dlib_pybind11 import find_peaks
from _dlib_pybind11 import find_projective_transform
from _dlib_pybind11 import full_object_detection
from _dlib_pybind11 import full_object_detections
from _dlib_pybind11 import function_evaluation
from _dlib_pybind11 import function_evaluation_request
from _dlib_pybind11 import function_spec
from _dlib_pybind11 import gaussian_blur
from _dlib_pybind11 import get_face_chip
from _dlib_pybind11 import get_face_chip_details
from _dlib_pybind11 import get_face_chips
from _dlib_pybind11 import get_frontal_face_detector
from _dlib_pybind11 import get_histogram
from _dlib_pybind11 import get_rect
from _dlib_pybind11 import global_function_search
from _dlib_pybind11 import grow_rect
from _dlib_pybind11 import hit_enter_to_continue
from _dlib_pybind11 import hough_transform
from _dlib_pybind11 import hysteresis_threshold
from _dlib_pybind11 import image_dataset_metadata
from _dlib_pybind11 import image_gradients
from _dlib_pybind11 import image_window
from _dlib_pybind11 import insert_image_chip
from _dlib_pybind11 import intersect
from _dlib_pybind11 import inv
from _dlib_pybind11 import jet
from _dlib_pybind11 import jitter_image
from _dlib_pybind11 import keyboard_mod_keys
from _dlib_pybind11 import label_connected_blobs
from _dlib_pybind11 import label_connected_blobs_watershed
from _dlib_pybind11 import length
from _dlib_pybind11 import line
from _dlib_pybind11 import load_grayscale_image
from _dlib_pybind11 import load_libsvm_formatted_data
from _dlib_pybind11 import load_rgb_alpha_image
from _dlib_pybind11 import load_rgb_image
from _dlib_pybind11 import make_bounding_box_regression_training_data
from _dlib_pybind11 import make_sparse_vector
from _dlib_pybind11 import matrix
from _dlib_pybind11 import max_cost_assignment
from _dlib_pybind11 import max_index_plus_one
from _dlib_pybind11 import max_point
from _dlib_pybind11 import max_point_interpolated
from _dlib_pybind11 import min_barrier_distance
from _dlib_pybind11 import mmod_rectangle
from _dlib_pybind11 import mmod_rectangles
from _dlib_pybind11 import mmod_rectangless
from _dlib_pybind11 import momentum_filter
from _dlib_pybind11 import no_convex_quadrilateral
from _dlib_pybind11 import non_printable_keyboard_keys
from _dlib_pybind11 import normalize_image_gradients
from _dlib_pybind11 import num_separable_filters
from _dlib_pybind11 import pair
from _dlib_pybind11 import partition_pixels
from _dlib_pybind11 import point
from _dlib_pybind11 import point_transform_projective
from _dlib_pybind11 import points
from _dlib_pybind11 import polygon_area
from _dlib_pybind11 import probability_that_sequence_is_increasing
from _dlib_pybind11 import pyramid_down
from _dlib_pybind11 import randomly_color_image
from _dlib_pybind11 import range
from _dlib_pybind11 import ranges
from _dlib_pybind11 import rangess
from _dlib_pybind11 import ranking_pair
from _dlib_pybind11 import ranking_pairs
from _dlib_pybind11 import rect_filter
from _dlib_pybind11 import rectangle
from _dlib_pybind11 import rectangles
from _dlib_pybind11 import rectangless
from _dlib_pybind11 import reduce
from _dlib_pybind11 import remove_incoherent_edge_pixels
from _dlib_pybind11 import resize_image
from _dlib_pybind11 import reverse
from _dlib_pybind11 import rgb_pixel
from _dlib_pybind11 import rvm_trainer_histogram_intersection
from _dlib_pybind11 import rvm_trainer_linear
from _dlib_pybind11 import rvm_trainer_radial_basis
from _dlib_pybind11 import rvm_trainer_sparse_histogram_intersection
from _dlib_pybind11 import rvm_trainer_sparse_linear
from _dlib_pybind11 import rvm_trainer_sparse_radial_basis
from _dlib_pybind11 import save_face_chip
from _dlib_pybind11 import save_face_chips
from _dlib_pybind11 import save_image
from _dlib_pybind11 import save_libsvm_formatted_data
from _dlib_pybind11 import scale_rect
from _dlib_pybind11 import segmenter_params
from _dlib_pybind11 import segmenter_test
from _dlib_pybind11 import segmenter_type
from _dlib_pybind11 import set_dnn_prefer_smallest_algorithms
from _dlib_pybind11 import shape_predictor
from _dlib_pybind11 import shape_predictor_training_options
from _dlib_pybind11 import shrink_rect
from _dlib_pybind11 import signed_distance_to_line
from _dlib_pybind11 import simple_object_detector
from _dlib_pybind11 import simple_object_detector_training_options
from _dlib_pybind11 import simple_test_results
from _dlib_pybind11 import skeleton
from _dlib_pybind11 import sobel_edge_detector
from _dlib_pybind11 import solve_structural_svm_problem
from _dlib_pybind11 import sparse_ranking_pair
from _dlib_pybind11 import sparse_ranking_pairs
from _dlib_pybind11 import sparse_vector
from _dlib_pybind11 import sparse_vectors
from _dlib_pybind11 import sparse_vectorss
from _dlib_pybind11 import spatially_filter_image
from _dlib_pybind11 import spatially_filter_image_separable
from _dlib_pybind11 import sub_image
from _dlib_pybind11 import suppress_non_maximum_edges
from _dlib_pybind11 import svm_c_trainer_histogram_intersection
from _dlib_pybind11 import svm_c_trainer_linear
from _dlib_pybind11 import svm_c_trainer_radial_basis
from _dlib_pybind11 import svm_c_trainer_sparse_histogram_intersection
from _dlib_pybind11 import svm_c_trainer_sparse_linear
from _dlib_pybind11 import svm_c_trainer_sparse_radial_basis
from _dlib_pybind11 import svm_rank_trainer
from _dlib_pybind11 import svm_rank_trainer_sparse
from _dlib_pybind11 import test_binary_decision_function
from _dlib_pybind11 import test_ranking_function
from _dlib_pybind11 import test_regression_function
from _dlib_pybind11 import test_sequence_segmenter
from _dlib_pybind11 import test_shape_predictor
from _dlib_pybind11 import test_simple_object_detector
from _dlib_pybind11 import threshold_filter_singular_values
from _dlib_pybind11 import threshold_image
from _dlib_pybind11 import tile_images
from _dlib_pybind11 import train_sequence_segmenter
from _dlib_pybind11 import train_shape_predictor
from _dlib_pybind11 import train_simple_object_detector
from _dlib_pybind11 import transform_image
from _dlib_pybind11 import translate_rect
from _dlib_pybind11 import vector
from _dlib_pybind11 import vectors
from _dlib_pybind11 import vectorss
from _dlib_pybind11 import zero_border_pixels
__all__: list[str] = ['DLIB_USE_BLAS', 'DLIB_USE_CUDA', 'DLIB_USE_LAPACK', 'KBD_MOD_ALT', 'KBD_MOD_CAPS_LOCK', 'KBD_MOD_CONTROL', 'KBD_MOD_META', 'KBD_MOD_NONE', 'KBD_MOD_NUM_LOCK', 'KBD_MOD_SCROLL_LOCK', 'KBD_MOD_SHIFT', 'KEY_ALT', 'KEY_BACKSPACE', 'KEY_CAPS_LOCK', 'KEY_CTRL', 'KEY_DELETE', 'KEY_DOWN', 'KEY_END', 'KEY_ESC', 'KEY_F1', 'KEY_F10', 'KEY_F11', 'KEY_F12', 'KEY_F2', 'KEY_F3', 'KEY_F4', 'KEY_F5', 'KEY_F6', 'KEY_F7', 'KEY_F8', 'KEY_F9', 'KEY_HOME', 'KEY_INSERT', 'KEY_LEFT', 'KEY_PAGE_DOWN', 'KEY_PAGE_UP', 'KEY_PAUSE', 'KEY_RIGHT', 'KEY_SCROLL_LOCK', 'KEY_SHIFT', 'KEY_UP', 'USE_AVX_INSTRUCTIONS', 'USE_NEON_INSTRUCTIONS', 'add_lib_to_dll_path', 'angle_between_lines', 'apply_cca_transform', 'array', 'as_grayscale', 'assignment_cost', 'auto_train_rbf_classifier', 'bottom_up_clustering', 'cca', 'cca_outputs', 'center', 'centered_rect', 'centered_rects', 'chinese_whispers', 'chinese_whispers_clustering', 'chip_details', 'chip_detailss', 'chip_dims', 'cnn_face_detection_model_v1', 'convert_image', 'convert_image_scaled', 'correlation_tracker', 'count_points_between_lines', 'count_points_on_side_of_line', 'count_steps_without_decrease', 'count_steps_without_decrease_robust', 'cross_validate_ranking_trainer', 'cross_validate_sequence_segmenter', 'cross_validate_trainer', 'cross_validate_trainer_threaded', 'cuda', 'distance_to_line', 'dot', 'dpoint', 'dpoints', 'drectangle', 'equalize_histogram', 'extract_image_4points', 'extract_image_chip', 'extract_image_chips', 'face_recognition_model_v1', 'fhog_object_detector', 'find_bright_keypoints', 'find_bright_lines', 'find_candidate_object_locations', 'find_dark_keypoints', 'find_dark_lines', 'find_line_endpoints', 'find_max_global', 'find_min_global', 'find_optimal_momentum_filter', 'find_optimal_rect_filter', 'find_peaks', 'find_projective_transform', 'full_object_detection', 'full_object_detections', 'function_evaluation', 'function_evaluation_request', 'function_spec', 'gaussian_blur', 'get_face_chip', 'get_face_chip_details', 'get_face_chips', 'get_frontal_face_detector', 'get_histogram', 'get_rect', 'global_function_search', 'grow_rect', 'hit_enter_to_continue', 'hough_transform', 'hysteresis_threshold', 'image_dataset_metadata', 'image_gradients', 'image_window', 'insert_image_chip', 'intersect', 'inv', 'jet', 'jitter_image', 'keyboard_mod_keys', 'label_connected_blobs', 'label_connected_blobs_watershed', 'length', 'line', 'load_grayscale_image', 'load_libsvm_formatted_data', 'load_rgb_alpha_image', 'load_rgb_image', 'make_bounding_box_regression_training_data', 'make_sparse_vector', 'matrix', 'max_cost_assignment', 'max_index_plus_one', 'max_point', 'max_point_interpolated', 'min_barrier_distance', 'mmod_rectangle', 'mmod_rectangles', 'mmod_rectangless', 'momentum_filter', 'no_convex_quadrilateral', 'non_printable_keyboard_keys', 'normalize_image_gradients', 'num_separable_filters', 'pair', 'partition_pixels', 'point', 'point_transform_projective', 'points', 'polygon_area', 'probability_that_sequence_is_increasing', 'pyramid_down', 'randomly_color_image', 'range', 'ranges', 'rangess', 'ranking_pair', 'ranking_pairs', 'rect_filter', 'rectangle', 'rectangles', 'rectangless', 'reduce', 'remove_incoherent_edge_pixels', 'resize_image', 'reverse', 'rgb_pixel', 'rvm_trainer_histogram_intersection', 'rvm_trainer_linear', 'rvm_trainer_radial_basis', 'rvm_trainer_sparse_histogram_intersection', 'rvm_trainer_sparse_linear', 'rvm_trainer_sparse_radial_basis', 'save_face_chip', 'save_face_chips', 'save_image', 'save_libsvm_formatted_data', 'scale_rect', 'segmenter_params', 'segmenter_test', 'segmenter_type', 'set_dnn_prefer_smallest_algorithms', 'shape_predictor', 'shape_predictor_training_options', 'shrink_rect', 'signed_distance_to_line', 'simple_object_detector', 'simple_object_detector_training_options', 'simple_test_results', 'skeleton', 'sobel_edge_detector', 'solve_structural_svm_problem', 'sparse_ranking_pair', 'sparse_ranking_pairs', 'sparse_vector', 'sparse_vectors', 'sparse_vectorss', 'spatially_filter_image', 'spatially_filter_image_separable', 'sub_image', 'suppress_non_maximum_edges', 'svm_c_trainer_histogram_intersection', 'svm_c_trainer_linear', 'svm_c_trainer_radial_basis', 'svm_c_trainer_sparse_histogram_intersection', 'svm_c_trainer_sparse_linear', 'svm_c_trainer_sparse_radial_basis', 'svm_rank_trainer', 'svm_rank_trainer_sparse', 'test_binary_decision_function', 'test_ranking_function', 'test_regression_function', 'test_sequence_segmenter', 'test_shape_predictor', 'test_simple_object_detector', 'threshold_filter_singular_values', 'threshold_image', 'tile_images', 'train_sequence_segmenter', 'train_shape_predictor', 'train_simple_object_detector', 'transform_image', 'translate_rect', 'vector', 'vectors', 'vectorss', 'zero_border_pixels']
def add_lib_to_dll_path(path):
    """
     On windows you must call os.add_dll_directory() to allow linking to external DLLs.  See
        https://docs.python.org/3.8/whatsnew/3.8.html#bpo-36085-whatsnew.  This function adds the folder
        containing path to the dll search path. 
        
    """
DLIB_USE_BLAS: bool = True
DLIB_USE_CUDA: bool = False
DLIB_USE_LAPACK: bool = True
KBD_MOD_ALT: _dlib_pybind11.keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_ALT: 4>
KBD_MOD_CAPS_LOCK: _dlib_pybind11.keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_CAPS_LOCK: 16>
KBD_MOD_CONTROL: _dlib_pybind11.keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_CONTROL: 2>
KBD_MOD_META: _dlib_pybind11.keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_META: 8>
KBD_MOD_NONE: _dlib_pybind11.keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_NONE: 0>
KBD_MOD_NUM_LOCK: _dlib_pybind11.keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_NUM_LOCK: 32>
KBD_MOD_SCROLL_LOCK: _dlib_pybind11.keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_SCROLL_LOCK: 64>
KBD_MOD_SHIFT: _dlib_pybind11.keyboard_mod_keys  # value = <keyboard_mod_keys.KBD_MOD_SHIFT: 1>
KEY_ALT: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_ALT: 3>
KEY_BACKSPACE: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_BACKSPACE: 0>
KEY_CAPS_LOCK: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_CAPS_LOCK: 5>
KEY_CTRL: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_CTRL: 2>
KEY_DELETE: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_DELETE: 16>
KEY_DOWN: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_DOWN: 14>
KEY_END: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_END: 9>
KEY_ESC: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_ESC: 6>
KEY_F1: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F1: 18>
KEY_F10: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F10: 27>
KEY_F11: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F11: 28>
KEY_F12: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F12: 29>
KEY_F2: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F2: 19>
KEY_F3: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F3: 20>
KEY_F4: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F4: 21>
KEY_F5: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F5: 22>
KEY_F6: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F6: 23>
KEY_F7: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F7: 24>
KEY_F8: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F8: 25>
KEY_F9: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_F9: 26>
KEY_HOME: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_HOME: 10>
KEY_INSERT: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_INSERT: 15>
KEY_LEFT: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_LEFT: 11>
KEY_PAGE_DOWN: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_PAGE_DOWN: 8>
KEY_PAGE_UP: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_PAGE_UP: 7>
KEY_PAUSE: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_PAUSE: 4>
KEY_RIGHT: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_RIGHT: 12>
KEY_SCROLL_LOCK: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_SCROLL_LOCK: 17>
KEY_SHIFT: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_SHIFT: 1>
KEY_UP: _dlib_pybind11.non_printable_keyboard_keys  # value = <non_printable_keyboard_keys.KEY_UP: 13>
USE_AVX_INSTRUCTIONS: bool = False
USE_NEON_INSTRUCTIONS: bool = True
__time_compiled__: str = 'Aug  4 2026 11:41:15'
__version__: str = '20.0.1'
