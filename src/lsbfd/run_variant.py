"""Run a pdm-bench pipeline with only some PU channels kept.

    python -m lsbfd.run_variant --channels vibration_1 ml <hydra overrides...>
    python -m lsbfd.run_variant --channels vibration_1 dl <hydra overrides...>

pdm-bench's PU loader keeps every HostService channel (two motor currents and
vibration) and has no option to drop any. This wraps the loader so the chosen channels
are the only ones a model sees, then hands over to the unchanged pipeline.
"""

from __future__ import annotations

import runpy
import sys


def keep_channels(channels: set[str]) -> None:
    from pdm_bench.loaders import pu

    original = pu._extract_signals_by_raster

    def filtered(mat_record):
        signals, times = original(mat_record)
        host = signals.get("HostService")
        if host is not None:
            missing = channels - host.keys()
            if missing:
                raise KeyError(f"PU HostService has no channel(s): {sorted(missing)}")
            signals["HostService"] = {k: v for k, v in host.items() if k in channels}
        return signals, times

    pu._extract_signals_by_raster = filtered


def main(argv: list[str]) -> None:
    if len(argv) < 3 or argv[0] != "--channels" or argv[2] not in {"ml", "dl"}:
        raise SystemExit("usage: python -m lsbfd.run_variant --channels a,b ml|dl [overrides]")
    keep_channels(set(argv[1].split(",")))
    module = f"pdm_bench.pipelines.{argv[2]}.pipeline"
    sys.argv = [module, *argv[3:]]
    # Run as __main__, exactly like `python -m <module>`, so Hydra finds its configs.
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
