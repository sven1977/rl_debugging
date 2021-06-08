import numpy as np

# From file mini_rllib.py in this directory.
from mini_rllib import SampleCollector

# Generate two sample collectors, one sequential and one parallel one.
sample_collector_sequential = SampleCollector(num_jobs=1)
sample_collector_parallel = SampleCollector(num_jobs=3)

# Sample from both and compare the results. Since the seeds are the same,
# they should both yield the exact same results.
samples1 = sample_collector_sequential.sample(num_episodes=3, seed=1234)
samples2 = sample_collector_parallel.sample(num_episodes=3, seed=1234)

# ... but the below assert fails :(
np.testing.assert_allclose(samples1, samples2)
