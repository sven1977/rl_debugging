**SETUP**

```
pip install gymnasium
git clone https://github.com/sven1977/rl_debugging
cd rl_debugging
python buggy_script.py
```

**INSTRUCTIONS**

The `buggy_script.py` file uses the SampleCollector class from `mini_rllib.py`
(all located in this root directory here) to collect observation samples
in parallel from n RL environments or in sequence from a single environment.

Each observation is a simple int for simplicity (discrete observation space).
Results from the different episodes are concatenated and returned.

However, using one parallel and one sequential setup of `SampleCollector` yields
different results (which should not be the case), even when
`SampleCollector.sample()` is provided with the same initial random seed.

Try to fix the assert at the end of `buggy_script.py` by finding the bug(s).
