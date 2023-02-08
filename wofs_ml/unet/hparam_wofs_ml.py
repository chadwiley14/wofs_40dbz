# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Write sample summary data for the hparams plugin.
See also `hparams_minimal_demo.py` in this directory for a demo that
runs much faster, using synthetic data instead of actually training
MNIST models.
"""

#GRAB GPU0
import py3nvml
py3nvml.grab_gpus(num_gpus=1, gpu_select=[0])

import os.path
import random
import shutil
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import xarray as xr 
from tensorboard.plugins.hparams import api as hp
import keras
from keras import Sequential
import sys
import matplotlib.pyplot as plt
import io
import scipy
from scipy.ndimage import gaussian_filter
import glob

# import tensorflow_probability as tfp

# GPU check
physical_devices = tf.config.list_physical_devices('GPU') 
n_physical_devices = len(physical_devices)
if(n_physical_devices > 0):
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
            
print('We have %d GPUs\n'%n_physical_devices)


if int(tf.__version__.split(".")[0]) < 2:
    # The tag names emitted for Keras metrics changed from "acc" (in 1.x)
    # to "accuracy" (in 2.x), so this demo does not work properly in
    # TensorFlow 1.x (even with `tf.enable_eager_execution()`).
    raise ImportError("TensorFlow 2.x is required to run this demo.")


flags.DEFINE_integer(
    "num_session_groups",
    100,
    "The approximate number of session groups to create.",
)

flags.DEFINE_string(
    "logdir",
    "/tmp/hparams_demo",
    "The directory to write the summary information to.",
)
flags.DEFINE_integer(
    "summary_freq",
    600,
    "Summaries will be written every n steps, where n is the value of "
    "this flag.",
)
flags.DEFINE_integer(
    "num_epochs",
    100,
    "Number of epochs per trial.",
)


INPUT_SHAPE = (128,128,14)
OUTPUT_CLASSES = 1

#convolution params
HP_CONV_LAYERS = hp.HParam("conv_layers", hp.IntInterval(1, 3))
HP_CONV_KERNEL_SIZE = hp.HParam("conv_kernel_size", hp.Discrete([3, 5, 7]))
HP_CONV_ACTIVATION = hp.HParam("conv_activation", hp.Discrete(['ELU', 'LeakyReLU', 'ReLU']))
HP_CONV_KERNELS = hp.HParam('num_of_kernels', hp.Discrete([4,8,16,32]))


#unet param
HP_UNET_DEPTH = hp.HParam('depth_of_unet', hp.Discrete([2,3,4,6]))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "rmsprop"]))
HP_LOSS = hp.HParam("loss", hp.Discrete(['weighted_binary_crossentropy'])) 
HP_BATCHNORM = hp.HParam('batchnorm', hp.Discrete([False, True]))
HP_BATCHSIZE = hp.HParam('batch_size', hp.Discrete([32,64,128,256,512]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-2,1e-3,1e-4, 1e-5]))
HP_LOSS_WEIGHTS = hp.HParam('loss_weights', hp.Discrete([1.0,2.0,3.0,4.0,5.0,6.0,7.0]))

HPARAMS = [HP_CONV_LAYERS,
    HP_CONV_KERNEL_SIZE,
    HP_CONV_ACTIVATION,
    HP_CONV_KERNELS,
    HP_UNET_DEPTH,
    HP_OPTIMIZER,
    HP_LOSS,
    HP_BATCHNORM,
    HP_BATCHSIZE,
    HP_LEARNING_RATE,
    HP_LOSS_WEIGHTS
]

METRICS = [
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val.)",
    ),
    hp.Metric(
        "epoch_loss",
        group="train",
        display_name="loss (train)",
    ),
    hp.Metric(
        "epoch_max_csi",
        group="validation",
        display_name="Max CSI (val.)",
    ),
    hp.Metric(
        "epoch_max_csi",
        group="train",
        display_name="Max CSI (train)",
    ),
    hp.Metric(
        "epoch_binary_accuracy",
        group="train",
        display_name="Binary Accuracy (train)",
    ),
    hp.Metric(
        "epoch_binary_accuracy",
        group="validation",
        display_name="Binary Accuracy (val)",
    ),
    hp.Metric(
        "epoch_auc",
        group = 'train',
        display_name='Epoch AUC'
    ),
    hp.Metric(
        'epoch_auc',
        group = "validation",
        display_name='Val. AUC'
    )
]

def build_loss_dict(weight):
    from custom_metrics_Chad import WeightedBinaryCrossEntropy
    loss_dict = {}
    loss_dict['binary_crossentropy'] = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss_dict['weighted_binary_crossentropy'] = WeightedBinaryCrossEntropy(weights=[weight,1.0])
    return loss_dict

def build_opt_dict(learning_rate):
    opt_dict = {}
    opt_dict['adam'] = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    opt_dict['adagrad'] = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    opt_dict['sgd'] = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    opt_dict['rmsprop'] = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    return opt_dict

def model_fn(hparams, seed):
    """Create a Keras model with the given hyperparameters.
    Args:
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
      seed: A hashable object to be used as a random seed (e.g., to
        construct dropout layers in the model).
    Returns:
      A compiled Keras model.
    """
    from keras_unet_collection import models

    rng = random.Random(seed)

    kernel_list = []
    for i in np.arange(1,hparams[HP_UNET_DEPTH]+1,1):
        kernel_list.append(hparams[HP_CONV_KERNELS]*i)

    model = models.unet_2d(INPUT_SHAPE, kernel_list, n_labels=OUTPUT_CLASSES,kernel_size=hparams[HP_CONV_KERNEL_SIZE],
                      stack_num_down=hparams[HP_CONV_LAYERS], stack_num_up=hparams[HP_CONV_LAYERS],
                      activation=hparams[HP_CONV_ACTIVATION], output_activation='Sigmoid', weights=None,
                      batch_norm=hparams[HP_BATCHNORM], pool='max', unpool='nearest', name='unet', l2=0.001)

    #get metric 
    from custom_metrics_Chad import MaxCriticalSuccessIndex
    from custom_metrics_Chad import WeightedBinaryCrossEntropy

    #compile losses: 
    #loss_dict = build_loss_dict(hparams[HP_LOSS_WEIGHT],hparams[HP_LOSS_THRESH])
    opt_dict = build_opt_dict(hparams[HP_LEARNING_RATE])
    loss_dict = build_loss_dict(hparams[HP_LOSS_WEIGHTS])
    model.compile(
        optimizer=opt_dict[hparams[HP_OPTIMIZER]],
        metrics=[MaxCriticalSuccessIndex(), tf.keras.metrics.AUC(),
        tf.keras.metrics.BinaryAccuracy()], #add more metrics here if you want them tf.keras.metrics... loss_dict[hparams[HP_LOSS]]
        loss = loss_dict[hparams[HP_LOSS]])#tf.keras.losses.BinaryCrossentropy(from_logits=False),

    return model

def prepare_data():
    """ Load data """
    #grab tf ds
    train_ds = tf.data.experimental.load('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/tf_ds/training_2017-2018.tf')
    train_2019 = tf.data.experimental.load('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/tf_ds/training_2019.tf')

    val_ds = tf.data.experimental.load('/ourdisk/hpc/ai2es/chadwiley/patches/data_30_NEW/tf_ds/validation_2020.tf')

    #concat the two training together
    train_ds = train_ds.concatenate(train_2019)



    #bring into memory
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    return(train_ds,val_ds)
    # #load in the training data.
    # examples = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/examples_padding.nc')
    # labels = xr.load_dataset('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/labels_padding.nc')
    # examples['dz_cress']=examples['dz_cress'].fillna(np.min(examples['dz_cress']))
    
    # for n in range(np.size(labels['dz_cress_binary'],axis=0)):
    #     labels['dz_cress_binary'][n] = scipy.ndimage.maximum_filter(labels['dz_cress_binary'][n], 5)

    # examples = examples.to_array()
    # examples = examples.transpose("n_samples",...)
    # examples = examples.transpose(...,"variable")

    # labels = labels.to_array()
    # labels = labels.transpose("n_samples", ...)
    # labels = labels.transpose(...,"variable")

    # train_examples = examples[:32905,:,:,:]
    # train_labels = labels[:32905,:,:,:]#FIX THE BREAKING OF DATASETS

    # val_examples = examples[32906:37019,:,:,:]
    # val_labels = labels[32906:37019,:,:,:]

    # train_examples = train_examples.to_numpy()
    # train_labels = train_labels.to_numpy()

    # val_examples = val_examples.to_numpy()
    # val_labels = val_labels.to_numpy()

    # print(np.shape(train_examples))
    # print(np.shape(train_labels))
    # print(np.shape(val_examples))
    # print(np.shape(val_labels))

    # test_examples = examples[37020:,:,:,:]
    # test_labels = labels[37020:,:,:,:]

    # test_examples = test_examples.to_dataset(dim = 'variable')
    # test_labels = test_labels.to_dataset(dim = 'variable')

    # test_examples.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/2021_2017_test_ex.nc')
    # test_labels.to_netcdf('/ourdisk/hpc/ai2es/chadwiley/patches/data_padding/2021_2017_test_labels_max_filter.nc')


    # ds_train = tf.data.Dataset.from_tensor_slices((train_examples, train_labels)) 
    # ds_val = tf.data.Dataset.from_tensor_slices((val_examples, val_labels))
    
    return (ds_train, ds_val)

def run(data, base_logdir, session_id, hparams):
    """Run a training/validation session.
    Flags must have been parsed for this function to behave.
    Args:
      data: The data as loaded by `prepare_data()`.
      base_logdir: The top-level logdir to which to write summary data.
      session_id: A unique string ID for this session.
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
    """
    model = model_fn(hparams=hparams, seed=session_id)
    logdir = os.path.join(base_logdir, session_id)

    ds_train,ds_val = data

    #batch the training data accordingly !!!CHANGE 3870 to your SAMPLE SIZE !!!
    ds_train = ds_train.shuffle(ds_train.cardinality().numpy()).batch(hparams[HP_BATCHSIZE])

    #this batch is arbitrary, just needed so that you don't overwhelm RAM. 
    ds_val = ds_val.batch(512)

    callback = tf.keras.callbacks.TensorBoard(
        logdir,
        update_freq='epoch',
        profile_batch=0,  # workaround for issue #2084
    )

    hparams_callback = hp.KerasCallback(logdir, hparams)

    
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    #add images to board 
    print(model.summary())

    result = model.fit(ds_train,
        epochs=flags.FLAGS.num_epochs,
        shuffle=False,
        validation_data=ds_val,
        callbacks=[callback, hparams_callback,callback_es],verbose=1)

    #Save result if want to make your own loss curves
    

    #save trained model, need to build path first 
    split_dir = logdir.split('log_2d_wbc')
    right = split_dir[0][:-1] + split_dir[1]
    left = '/scratch/chadwiley/models_2d_wbc/'
    model.save(left + right + "model.h5")


def run_all(logdir, verbose=False):
    """Perform random search over the hyperparameter space.
    Arguments:
      logdir: The top-level directory into which to write data. This
        directory should be empty or nonexistent.
      verbose: If true, print out each run's name as it begins.
    """
    data = prepare_data()
    rng = random.Random(0)

    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)

    sessions_per_group = 1
    num_sessions = flags.FLAGS.num_session_groups * sessions_per_group
    session_index = 0  # across all session groups
    for group_index in range(flags.FLAGS.num_session_groups):
        hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
        hparams_string = str(hparams)
        for repeat_index in range(sessions_per_group):
            session_id = str(session_index)
            session_index += 1
            if verbose:
                print(
                    "--- Running training session %d/%d"
                    % (session_index, num_sessions)
                )
                print(hparams_string)
                print("--- repeat #: %d" % (repeat_index + 1))
            run(
                data=data,
                base_logdir=logdir,
                session_id=session_id,
                hparams=hparams,
            )


def main(unused_argv):
    np.random.seed(0)
    logdir = flags.FLAGS.logdir
    print('removing old logs')
    shutil.rmtree(logdir, ignore_errors=True)
    print("Saving output to %s." % logdir)
    run_all(logdir=logdir, verbose=True)
    print("Done. Output saved to %s." % logdir)


if __name__ == "__main__":
    app.run(main)
