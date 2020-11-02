"""
Multi node GPU based random forest using Nvidia RAPIDS
Adapted from https://github.com/rapidsai/cuml/blob/branch-0.13/notebooks/random_forest_mnmg_demo.ipynb
"""

import numpy as np
import sklearn
import time

import pandas as pd
import cudf
import cuml
import os

from sklearn.metrics import accuracy_score
from sklearn import model_selection, datasets

from cuml.dask.common import utils as dask_utils
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask_cudf

from cuml.dask.ensemble import RandomForestClassifier as cumlDaskRF
from sklearn.ensemble import RandomForestClassifier as sklRF

if __name__ == "__main__":
    ## using dask to setup cluster

    # This will use all GPUs on the local host by default
    # set this to use on node disk for caching
    cluster = LocalCUDACluster(local_directory=os.environ['TMPDIR'], threads_per_worker=1)
    c = Client(cluster)

    # Query the client for all connected workers
    workers = c.has_what().keys()
    n_workers = len(workers)
    n_streams = 8 # Performance optimization

    ## setting parameters

    # Data parameters
    train_size = 100000
    test_size = 1000
    n_samples = train_size + test_size
    n_features = 20

    # Random Forest building parameters
    max_depth = 12
    n_bins = 16
    n_trees = 1000

    ## generating some data
    # use sklearn API to generate some psuedo data to test this out
    # we do this on the client (initial process) before distributing this out to workers

    X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features,
                                    n_clusters_per_class=1, n_informative=int(n_features / 3),
                                    random_state=123, n_classes=5)
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)

    n_partitions = n_workers

    # First convert to cudf (with real data, you would likely load in cuDF format to start)
    X_train_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
    y_train_cudf = cudf.Series(y_train)

    # Partition with Dask
    # In this case, each worker will train on 1/n_partitions fraction of the data
    X_train_dask = dask_cudf.from_cudf(X_train_cudf, npartitions=n_partitions)
    y_train_dask = dask_cudf.from_cudf(y_train_cudf, npartitions=n_partitions)

    # Persist to cache the data in active memory
    X_train_dask, y_train_dask = \
      dask_utils.persist_across_workers(c, [X_train_dask, y_train_dask], workers=workers)

    # do this model fit with cpus using sklearn
    # Use all avilable CPU cores
    t0 = time.time()
    skl_model = sklRF(max_depth=max_depth, n_estimators=n_trees, n_jobs=-1)
    skl_model.fit(X_train, y_train)
    t1 = time.time()

    cpu_time = t1 - t0

    print(f"CPU model fit took: {cpu_time}.")

    ## fit model using Dask cuML
    t0 = time.time()
    cuml_model = cumlDaskRF(max_depth=max_depth, n_estimators=n_trees, n_bins=n_bins, n_streams=n_streams)
    cuml_model.fit(X_train_dask, y_train_dask, convert_dtype=True)

    wait(cuml_model.rfs) # Allow asynchronous training tasks to finish
    t1 = time.time()
    gpu_time = t1 - t0 

    print(f"cuML model fit time: {gpu_time}.")

    ## compare accuracy
    skl_y_pred = skl_model.predict(X_test)
    cuml_y_pred = cuml_model.predict(X_test)

    # Due to randomness in the algorithm, you may see slight variation in accuracies
    print("SKLearn accuracy:  ", accuracy_score(y_test, skl_y_pred))
    print("CuML accuracy:     ", accuracy_score(y_test, cuml_y_pred))
