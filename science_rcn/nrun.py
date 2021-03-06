"""
Reference implementation of a two-level RCN model for MNIST classification experiments.

Examples:
- To run a small unit test that trains and tests on 20 images using one CPU 
  (takes ~2 minutes, accuracy is ~60%):
python science_rcn/run.py

- To run a slightly more interesting experiment that trains on 100 images and tests on 20 
  images using multiple CPUs (takes <1 min using 7 CPUs, accuracy is ~90%):
python science_rcn/run.py --train_size 100 --test_size 20 --parallel

- To test on the full 10k MNIST test set, training on 1000 examples 
(could take hours depending on the number of available CPUs, average accuracy is ~97.7+%):
python science_rcn/run.py --full_test_set --train_size 1000 --parallel --pool_shape 25 --perturb_factor 2.0
"""

import argparse
import logging
import numpy as np
import os
from multiprocessing import Pool
from functools import partial
from scipy.misc import imresize
from scipy.ndimage import imread
import pickle
import time

import pdb

from science_rcn.inference import test_image
from science_rcn.learning import train_image

form_index = 300

LOG = logging.getLogger(__name__)

def run_test_image(data_dir='tmp',
                   model_file='default_model.pkl',
                   train_size=20,
                   test_size=20,
                   fit_level=1,
                   full_test_set=False,
                   pool_shape=(25, 25),
                   perturb_factor=2.,
                   parallel=True,
                   verbose=False,
                   seed=5,
                   class_count=10):
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    test_data = load_data(data_dir, None)
    if len(test_data) <= 0:
        print('No Image file in path %s'%data_dir)
        return 
    # Multiprocessing set up
    num_workers = None if parallel else 1
    pool = Pool(num_workers)
    all_model_factors = pickle.load(open(model_file, 'rb'))
    all_model_factors = zip(*all_model_factors)
    # train_results = pickle.load(open('train_results.pkl', 'rb'))

    LOG.info("Testing on {} images...".format(len(test_data)))
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    start_time = time.time()
    test_partial = partial(test_image, model_factors=all_model_factors,
                           pool_shape=pool_shape,
                           num_candidates=5,
                           n_iters=20,
                           test_size=test_size)
    test_results = pool.map_async(test_partial, [d for d in test_data]).get(9999999)

    # Evaluate result
    correct = 0
    for test_idx, (winner_idx, _) in enumerate(test_results):
        print(int(test_data[test_idx][1]), winner_idx, train_size, class_count, winner_idx // (train_size // class_count))
        correct += int(test_data[test_idx][1])%10 == winner_idx // (train_size // class_count)
    print "Testing use {} s".format(time.time()-start_time)
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    print "Total test accuracy = {}".format(float(correct) / len(test_results))

def run_test_only(data_dir='data/MNIST1',
                   model_file='default_model.pkl',
                   train_size=20,
                   test_size=20,
                   fit_level=1,
                   full_test_set=False,
                   pool_shape=(25, 25),
                   perturb_factor=2.,
                   parallel=True,
                   verbose=False,
                   seed=5,
                   class_count=10):
    """Run MNIST test evaluate results. 

    Parameters
    ----------
    data_dir : string
        Dataset directory.
    train_size, test_size : int
        MNIST dataset sizes are in increments of 10
    full_test_set : bool
        Test on the full MNIST 10k test set.
    pool_shape : (int, int)
        Vertical and horizontal pool shapes.
    perturb_factor : float
        How much two points are allowed to vary on average given the distance
        between them. See Sec S2.3.2 for details.
    parallel : bool
        Parallelize over multiple CPUs.
    verbose : bool
        Higher verbosity level.
    seed : int
        Random seed used by numpy.random for sampling training set.
    
    Returns
    -------
    model_factors : ([numpy.ndarray], [numpy.ndarray], [networkx.Graph])
        ([frcs], [edge_factors], [graphs]), outputs of train_image in learning.py.
    test_results : [(int, float)]
        List of (winner_idx, winner_score), outputs of test_image in inference.py.
    """
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    # Multiprocessing set up
    num_workers = None if parallel else 1
    pool = Pool(num_workers)

    train_data, test_data = get_mnist_data_iters(
        data_dir, train_size, test_size, full_test_set, seed=seed)
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    all_model_factors = pickle.load(open(model_file, 'rb'))
    # train_results = pickle.load(open('train_results.pkl', 'rb'))

    LOG.info("Testing on {} images...".format(len(test_data)))
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    start_time = time.time()
    # print(test_data)
    test_partial = partial(test_image, model_factors=all_model_factors,
                           pool_shape=pool_shape,
                           num_candidates=5,
                           n_iters=20,
                           test_size=test_size)
    test_results = pool.map_async(test_partial, [d for d in test_data]).get(9999999)

    # Evaluate result
    correct = 0
    for test_idx, (winner_idx, _) in enumerate(test_results):
        print(int(test_data[test_idx][1]), winner_idx, train_size, class_count, winner_idx // (train_size // class_count))
        correct += int(test_data[test_idx][1]) == winner_idx // (train_size // class_count)
    print "Testing use {} s".format(time.time()-start_time)
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    print "Total test accuracy = {}".format(float(correct) / len(test_results))

    return all_model_factors, test_results


def run_experiment(data_dir='data/MNIST1',
                   model_file='default_model.pkl',
                   train_size=20,
                   test_size=20,
                   fit_level=1,
                   full_test_set=False,
                   pool_shape=(25, 25),
                   perturb_factor=2.,
                   parallel=True,
                   verbose=False,
                   seed=5,
                   class_count=10):
    """Run MNIST experiments and evaluate results. 

    Parameters
    ----------
    data_dir : string
        Dataset directory.
    train_size, test_size : int
        MNIST dataset sizes are in increments of 10
    full_test_set : bool
        Test on the full MNIST 10k test set.
    pool_shape : (int, int)
        Vertical and horizontal pool shapes.
    perturb_factor : float
        How much two points are allowed to vary on average given the distance
        between them. See Sec S2.3.2 for details.
    parallel : bool
        Parallelize over multiple CPUs.
    verbose : bool
        Higher verbosity level.
    seed : int
        Random seed used by numpy.random for sampling training set.
    
    Returns
    -------
    model_factors : ([numpy.ndarray], [numpy.ndarray], [networkx.Graph])
        ([frcs], [edge_factors], [graphs]), outputs of train_image in learning.py.
    test_results : [(int, float)]
        List of (winner_idx, winner_score), outputs of test_image in inference.py.
    """
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    # Multiprocessing set up
    num_workers = None if parallel else 1
    pool = Pool(num_workers)

    # pdb.set_trace()
    train_data, test_data = get_mnist_data_iters(
        data_dir, train_size, test_size, full_test_set, seed=seed)

    LOG.info("Training on {} images...".format(len(train_data)))
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    start_time = time.time()
    train_partial = partial(train_image,
                            perturb_factor=perturb_factor)
    split = 100
    # train_data = train_data[0:34]
    train_results = []
    for i in range(form_index, 1+len(train_data)//split):
      t_data = train_data[i*split:i*split+split]
      # print(t_data)
      result = pool.map_async(train_partial, [d for d in t_data]).get(9999999)
      print(time.strftime('%H:%M:%S', time.localtime(time.time())))
      pickle.dump(result, open('temp_model/model_%d_%d.pkl'%(i*split,i*split+len(t_data)), 'wb'))
      train_results.extend(result)
    # train_results = []
    # for i in range(len(train_data)):
    #     train_results.append(train_partial(train_data[i], perturb_factor=perturb_factor))

    print "Training use {} ms".format(time.time()-start_time)
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))

    for i in range(len(train_results)):
        if not train_results[i]:
            train_results[i] = train_results[0]
    # pickle.dump(train_results, open('train_results.pkl', 'wb'))
    all_model_factors = zip(*train_results)

    pickle.dump(all_model_factors, open(model_file, 'wb'))

    # LOG.info("Testing on {} images...".format(len(test_data)))
    # start_time = time.time()
    # test_partial = partial(test_image, model_factors=all_model_factors,
    #                        pool_shape=pool_shape,
    #                        num_candidates=5,
    #                        n_iters=20,
    #                        test_size=test_size)
    # test_results = pool.map_async(test_partial, [d for d in test_data]).get(9999999)

    # # Evaluate result
    # correct = 0
    # for test_idx, (winner_idx, _) in enumerate(test_results):
    #     print(int(test_data[test_idx][1]), winner_idx, train_size, class_count, winner_idx // (train_size // class_count))
    #     correct += int(test_data[test_idx][1]) == winner_idx // (train_size // class_count)
    # print "Testing use {} s".format(time.time()-start_time)
    # print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    # print "Total test accuracy = {}".format(float(correct) / len(test_results))

    # return all_model_factors, test_results


def load_data(image_dir, num_per_class, get_filenames=False):
    loaded_data = []
    try:
        os.remove(os.path.join(image_dir, '.DS_Store'))
    except:
        pass
    for category in sorted(os.listdir(image_dir)):
        cat_path = os.path.join(image_dir, category)
        if not os.path.isdir(cat_path) or category.startswith('.'):
            continue
        try:
            os.remove(os.path.join(cat_path, '.DS_Store'))
        except:
            pass
        # print('cat_path %s'%cat_path)
        if num_per_class is None:
            samples = sorted(os.listdir(cat_path))
        else:
            samples = np.random.choice(sorted(os.listdir(cat_path)), num_per_class)


        for fname in samples:
            if len(loaded_data) < (form_index*100-1) or len(loaded_data)>(form_index*100+501):
                loaded_data.append(('a', 'b'))
                continue
            # print(fname)
            filepath = os.path.join(cat_path, fname)

            # Resize and pad the images to (200, 200)
            image_arr = imresize(imread(filepath), (112, 112))
            img = np.pad(image_arr,
                         pad_width=tuple([(p, p) for p in (44, 44)]),
                         mode='constant', constant_values=0)
            loaded_data.append((img, category))

    return loaded_data

def get_mnist_data_iters(data_dir, train_size, test_size,
                         full_test_set=False, seed=5, class_count=10):
    """
    Load MNIST data.

    Assumed data directory structure:
        training/
            0/
            1/
            2/
            ...
        testing/
            0/
            ...

    Parameters
    ----------
    train_size, test_size : int
        MNIST dataset sizes are in increments of 10
    full_test_set : bool
        Test on the full MNIST 10k test set.
    seed : int
        Random seed used by numpy.random for sampling training set.

    Returns
    -------
    train_data, train_data : [(numpy.ndarray, str)]
        Each item reps a data sample (2-tuple of image and label)
        Images are numpy.uint8 type [0,255]
    """
    if not os.path.isdir(data_dir):
        raise IOError("Can't find your data dir '{}'".format(data_dir))

    np.random.seed(seed)
    train_set = load_data(os.path.join(data_dir, 'training'),
                           num_per_class=train_size // class_count)
    test_set = []; #load_data(os.path.join(data_dir, 'testing'),
                   #       num_per_class=None if full_test_set else test_size // class_count)
    return train_set, test_set


if __name__ == '__main__':
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_size',
        dest='train_size',
        type=int,
        default=20,
        help="Number of training examples.",
    )
    parser.add_argument(
        '--test_size',
        dest='test_size',
        type=int,
        default=20,
        help="Number of testing examples.",
    )
    parser.add_argument(
        '--class_count',
        dest='class_count',
        type=int,
        default=10,
        help="Number of class count.",
    )
    parser.add_argument(
        '--full_test_set',
        dest='full_test_set',
        action='store_true',
        default=False,
        help="Test on full MNIST test set.",
    )
    parser.add_argument(
        '--pool_shapes',
        dest='pool_shape',
        type=int,
        default=25,
        help="Pool shape.",
    )
    parser.add_argument(
        '--perturb_factor',
        dest='perturb_factor',
        type=float,
        default=2.,
        help="Perturbation factor.",
    )
    parser.add_argument(
        '--fit_level',
        dest='fit_level',
        type=int,
        default=1,
        help="fit level big fit better.",
    )
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        default='data/MNIST',
        help="data input dir.",
    )
    parser.add_argument(
        '--model_file',
        dest='model_file',
        default='default_model.pkl',
        help="model filepath.",
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        default=5, #np.random.randint(1,30),
        help="Seed for numpy.random to sample training and testing dataset split.",
    )
    parser.add_argument(
        '--parallel',
        dest='parallel',
        default=False,
        action='store_true',
        help="Parallelize over multi-CPUs if True.",
    )
    parser.add_argument(
        '--verbose',
        dest='verbose',
        action='store_true',
        default=False,
        help="Verbosity level.",
    )
    parser.add_argument(
        '--test_only',
        dest='test_only',
        action='store_true',
        default=False,
        help="use saved model test only.",
    )
    parser.add_argument(
        '--test_image',
        dest='test_image',
        action='store_true',
        default=False,
        help="test images.",
    )
    options = parser.parse_args()

    if options.test_image:
        run_test_image(model_file=options.model_file,
                       train_size=options.train_size,
                       data_dir=options.data_dir,
                       test_size=options.test_size,
                       full_test_set=options.full_test_set,
                       pool_shape=(options.pool_shape, options.pool_shape),
                       perturb_factor=options.perturb_factor,
                       seed=options.seed,
                       verbose=options.verbose,
                       parallel=options.parallel,
                       class_count=options.class_count)
    elif options.test_only:
        # run_experiment(train_size=options.train_size,
        run_test_only(train_size=options.train_size,
                       model_file=options.model_file,
                       data_dir=options.data_dir,
                       test_size=options.test_size,
                       full_test_set=options.full_test_set,
                       pool_shape=(options.pool_shape, options.pool_shape),
                       perturb_factor=options.perturb_factor,
                       seed=options.seed,
                       verbose=options.verbose,
                       parallel=options.parallel,
                       class_count=options.class_count)
    else:
        run_experiment(train_size=options.train_size,
                       model_file=options.model_file,
                       data_dir=options.data_dir,
                       test_size=options.test_size,
                       full_test_set=options.full_test_set,
                       pool_shape=(options.pool_shape, options.pool_shape),
                       perturb_factor=options.perturb_factor,
                       seed=options.seed,
                       verbose=options.verbose,
                       parallel=options.parallel,
                       class_count=options.class_count)
