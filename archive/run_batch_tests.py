#!/usr/bin/env python3
"""
Test runner for batch metric assignment tests.

Run this script to verify that the batch metric evaluation correctly
assigns metrics to the right images, even when some images don't have
bounding boxes.

Usage:
    python run_batch_tests.py
    
    # Or run specific tests:
    python run_batch_tests.py --test test_batch_metric_assignment.py
    python run_batch_tests.py --test test_batch_metric_integration.py
"""

import sys
import argparse
import subprocess
from pathlib import Path


def run_tests(test_file: str = None, verbose: bool = True, stop_on_first_failure: bool = False):
    """Run the batch metric tests."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
        cmd.append("-s")  # Don't capture output (allows print statements to show)
    
    if stop_on_first_failure:
        cmd.append("-x")
    
    # Add specific test file or run all batch tests
    if test_file:
        if not test_file.startswith("tests/"):
            test_file = f"tests/{test_file}"
        cmd.append(test_file)
    else:
        # Run both test files
        cmd.extend([
            "tests/test_batch_metric_assignment.py",
            "tests/test_batch_metric_integration.py"
        ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("ERROR: pytest not found. Please install it with: pip install pytest")
        return 1
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 130


def main():
    parser = argparse.ArgumentParser(description="Run batch metric assignment tests")
    parser.add_argument(
        "--test", 
        help="Specific test file to run (e.g., test_batch_metric_assignment.py)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true", 
        help="Run in quiet mode (less verbose output)"
    )
    parser.add_argument(
        "--stop-on-fail", 
        action="store_true", 
        help="Stop on first test failure"
    )
    
    args = parser.parse_args()
    
    print("Batch Metric Assignment Test Runner")
    print("=" * 40)
    print()
    
    if args.test:
        print(f"Running specific test: {args.test}")
    else:
        print("Running all batch metric tests:")
        print("  - test_batch_metric_assignment.py")
        print("  - test_batch_metric_integration.py")
    
    print()
    
    # Check if test files exist
    test_files = []
    if args.test:
        test_file = args.test if args.test.startswith("tests/") else f"tests/{args.test}"
        test_files.append(test_file)
    else:
        test_files = [
            "tests/test_batch_metric_assignment.py",
            "tests/test_batch_metric_integration.py"
        ]
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"ERROR: Test file not found: {test_file}")
            return 1
    
    # Run the tests
    exit_code = run_tests(
        test_file=args.test,
        verbose=not args.quiet,
        stop_on_first_failure=args.stop_on_fail
    )
    
    print()
    print("=" * 80)
    if exit_code == 0:
        print("✅ All tests passed!")
        print()
        print("This means that batch metric evaluation correctly assigns")
        print("metrics to the right images, even when some images don't")
        print("have bounding boxes.")
    else:
        print("❌ Some tests failed!")
        print()
        print("This indicates there may be issues with how batch metrics")
        print("are assigned to images. Please check the test output above.")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())