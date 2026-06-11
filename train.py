"""
Orchestrator — runs all 3 training phases in sequence.
Usage: python train.py
       python train.py --start 2   (resume from phase 2)
"""

import argparse
import train_phase1
import train_phase2
import train_phase3

PHASES = {
    1: train_phase1.train,
    2: train_phase2.train,
    3: train_phase3.train,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, choices=[1, 2, 3],
                        help="Phase to start from (default: 1)")
    args = parser.parse_args()

    for phase in range(args.start, 4):
        print(f"\n{'='*50}")
        print(f"Starting Phase {phase}")
        print(f"{'='*50}\n")
        PHASES[phase]()

    print("\nAll phases complete.")
