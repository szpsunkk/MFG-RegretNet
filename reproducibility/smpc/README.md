# Secure computation benchmark

This is the separate two-party CrypTen benchmark of the final P2-v2 mechanism. The web platform executes its mechanism on Cloudflare in plaintext over HTTPS; it does not execute this protocol.

The recorded environment is **Python 3.8.3, torch 1.13.1+cpu, CrypTen 0.4.1**, 16 fractional bits, the reciprocal sigmoid/tanh implementation, trusted first-party preprocessing (`TFP`), loopback Gloo, and four threads per party. Use a separate environment from the FL package. The original machine was an Intel Xeon Silver 4316; measured times are machine dependent and do not represent WAN execution or a fully deployed independent-party service.

```bash
python3.8 -m venv .venv-smpc
source .venv-smpc/bin/activate
pip install torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install 'numpy<2' scipy omegaconf scikit-learn
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install crypten==0.4.1

cd reproducibility/smpc
python test_integer_counters.py
python test_mechanism.py
python bench.py --n 10 --screening --repetitions 1 --out validation/screen_N10.json
```

Screen N=10,50,100,200,500 in ascending order; stop if a size exceeds the declared limits. For a passing size, run `bench.py --n N --repetitions 5 --out validation/formal_NN.json`. Preserve all failures. The program asserts the exact major environment choices and records the arithmetic errors, communication, time, and the purchase-total diagnostic. `primitive_bench.py --help` documents the per-operation measurement interface; `recipes/` gives the exact public operation counts/shapes for cost estimates.

The final results are in `smpc_p2v2.json`. The **A2 screening limits were changed after earlier failures** to allocation/payment error 0.1 and dividend error 0.05; this is disclosed in the result and `history/`. They are screening tolerances, not proofs of exact secure equivalence. Earlier failures and the reference operation graph are retained. The numerical algorithm was not changed to fit the thresholds.

The original frozen reference files and checkpoint are checked by `reference_manifest.json`. The public manifest omits internal task-instruction files; numerical source and checkpoint hashes are identical. `support.py` and the numerical benchmark scripts are unchanged.
