from .data_augmentation import augmentation
from tensorflow.data import TFRecordDataset, AUTOTUNE, Dataset
from tensorflow.io import FixedLenFeature, parse_single_example
from .constants import IMAGE_SHAPE

def datasets_from_tfrecords(records_ref, records_src, buf_size_ref, buf_sizes_src, batch_size=1, compression_type='GZIP'):
    """
    Creates Tensorflow datasets from Tensorflow records paths without bias sampling. The shape of the MR images must correspond to the IMAGE_SHAPE gloabl variable.
    INPUTS :
      - records_ref : one or several tfrecord filepaths for the reference dataset (str or list of str).
      - records_src : one or several tfrecord filepaths for each source dataset (list of str or list of list of str)
      - buf_size_ref : buffer_size for the shuffle of the reference dataset (integer)
      - buf_size_src : buffer_size for the shuffle of each source dataset (list of integers)
      - batch_size
      - compression_type : see https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
    RETURNS :
      - a list of infinite Tensorflow dataset iterators. Each one is associated to a source dataset and yields one batch with images from the reference dataset and images from the source dataset
    """
    keys_to_feature = {'mri': FixedLenFeature(shape=IMAGE_SHAPE, dtype='float32')}
    dsRef = TFRecordDataset(records_ref, compression_type=compression_type, num_parallel_reads=AUTOTUNE)
    dsRef = dsRef.shuffle(buf_size_ref)
    
    dsSrcs = [TFRecordDataset(records, compression_type=compression_type, num_parallel_reads=AUTOTUNE).shuffle(buf_size)
                 for records,buf_size in zip(records_src, buf_sizes_src)]
    datasets = [Dataset.zip(dsRef, ds).repeat() for ds in dsSrcs]
    datasets = [ds.map(lambda t1, t2: (augmentation(parse_single_example(t1, keys_to_feature)['mri']),
                                       augmentation(parse_single_example(t2, keys_to_feature)['mri'])),
                       num_parallel_calls=AUTOTUNE, deterministic=False) for ds in datasets]
    return [ds.batch(batch_size, num_parallel_calls=AUTOTUNE, deterministic=False).prefetch(AUTOTUNE).__iter__() for ds in datasets]
    
    
    
def datasets_from_tfrecords_biasSampling(records_entries, batch_size=1, compression_type='GZIP'):
    """
    Creates Tensorflow datasets from Tensorflow records with bias sampling. The shape of the MR images must correspond to the IMAGE_SHAPE global variable.
    INPUTS : 
        - records_entries : for each source site, a tuple with the following elements :
            1. list of tfrecord filepaths (or list of list) for the source site
            2. list of tfrecord filepaths (or list of list) for the reference site
            3. list of sampling probabilities (associated with each element in 1 and 2)
            4. buffer size associated to each reference tfrecords element (list of integers)
            5. buffer size associated to each source tfrecords element (list of integers)
        - batch_size
        - compression_type
    RETURNS :
        - a list of infinite Tensorflow dataset iterators. Each one is associated to a source dataset and yields one batch with images from the reference dataset and images from the source dataset
    """
    keys_to_feature = {'mri': FixedLenFeature(shape=IMAGE_SHAPE, dtype='float32')}
    def datasetZip_from_pairs(tfrecordsA, tfrecordsB, probasSampling, buf_sizesA, buf_sizesB):
        def datasetList_from_tfrecords(tfrecords, buf_sizes):
            return [TFRecordDataset(tfr, num_parallel_reads=AUTOTUNE, compression_type=compression_type).shuffle(b_size).repeat() for tfr,b_size in zip(tfrecords, buf_sizes)]
    
        datasetsA = datasetList_from_tfrecords(tfrecordsA, buf_sizesA)
        datasetsB = datasetList_from_tfrecords(tfrecordsB, buf_sizesB)
        dsZip_list = [Dataset.zip((dsA,dsB)) for dsA,dsB in zip(datasetsA,datasetsB)]
        dsZip = Dataset.sample_from_datasets(dsZip_list, weights=probasSampling, rerandomize_each_iteration=True)
        dsZip = dsZip.map(lambda t1,t2: (augmentation(parse_single_example(t1, keys_to_feature)['mri']),
                                         augmentation(parse_single_example(t2, keys_to_feature)['mri'])),
                          num_parallel_calls=AUTOTUNE, deterministic=False)
        return dsZip.batch(batch_size, num_parallel_calls=AUTOTUNE, deterministic=False).prefetch(AUTOTUNE)
    
    return [datasetZip_from_pairs(*args).__iter__() for args in records_entries]