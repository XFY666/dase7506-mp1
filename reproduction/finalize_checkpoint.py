"""Turn a finished training run into the final CPU predictor checkpoint.

Two steps, both of which leave every model tensor unchanged:

  ema       copy the EMA weights stored in train_experiment.py's final
            resume.pt into an ordinary student checkpoint. The trainer's own
            checkpoint.pt keeps its native-best step, which is not the model
            that was selected.
  schedule  add the CPU scheduling keys used for the timed submission
            (8 windows per Transformer batch, 4 per vocabulary chunk,
            restructured count mixing) to a hybrid checkpoint written by
            tune_hybrid.py.

    python reproduction/finalize_checkpoint.py ema --input RUNS/final-32k/resume.pt \
        --output RUNS/final-ema/checkpoint.pt
    python reproduction/finalize_checkpoint.py schedule --input RUNS/final-hybrid/checkpoint.pt \
        --output RUNS/final/checkpoint.pt
"""
import argparse
import hashlib
from pathlib import Path

import torch

RESUME_ONLY = ('optimizer', 'ema', 'sampling_rng', 'torch_rng', 'cuda_rng')
SCHEDULE = dict(cpu_batch_chunk=4, cpu_trunk_chunk=8, cpu_restructure_counts=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['ema', 'schedule'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    source = torch.load(args.input, map_location='cpu', weights_only=True)
    if args.mode == 'ema':
        if source.get('implementation') != 'student' or source.get('ema') is None:
            raise ValueError('Expected a train_experiment.py resume.pt that contains EMA weights.')
        result = {key: value for key, value in source.items() if key not in RESUME_ONLY}
        result.update(model=source['ema'], weight_kind='ema', validation=None,
                      endpoint=dict(source_resume_sha256=sha(args.input), step=source['step']))
    else:
        if source.get('implementation') != 'hybrid':
            raise ValueError('Expected a hybrid checkpoint written by tune_hybrid.py.')
        result = dict(source, config=dict(source['config'], **SCHEDULE))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(args.output, sha(args.output))


if __name__ == '__main__':
    main()
