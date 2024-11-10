#!/usr/bin/env python3
import subprocess
import argparse
import sys
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class LeanWorkDir:
    """Stores working directory information for the Lean project"""
    base_dir: Path
    project_name: str
    project_dir: Path
    test_dir: Path
    repl_dir: Path
    repl_binary: Path

    @classmethod
    def create(cls, project_name: str, base_dir: str | None = None) -> 'LeanWorkDir':
        base = Path(base_dir) if base_dir else Path.cwd()
        project_dir = base / project_name
        test_dir = project_dir / "test"
        repl_dir = test_dir / "repl"
        repl_binary = repl_dir / ".lake" / "build" / "bin" / "repl"
        return cls(base, project_name, project_dir, test_dir, repl_dir, repl_binary)

def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    silent: bool = False,
    stdin_input: str | None = None
) -> tuple[bool, str, str]:
    """Run a command and return success status and output."""
    try:
        process = subprocess.run(
            cmd,
            check=True,
            cwd=cwd,
            input=stdin_input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if not silent:
            if process.stdout:
                print(process.stdout)
            if process.stderr:
                print(process.stderr, file=sys.stderr)
        return True, process.stdout, process.stderr
    except subprocess.CalledProcessError as e:
        if not silent:
            print(f"Error running command {' '.join(cmd)}:")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
        return False, e.stdout, e.stderr

def check_elan_version() -> bool:
    """Check if elan is installed and up to date."""
    success, stdout, _ = run_command(['elan', '--version'], silent=True)
    if not success:
        print("Error: elan is not installed. Please install elan first.")
        print("Visit: https://github.com/leanprover/elan#installation")
        return False

    try:
        version = stdout.split()[1]
        major = int(version.split('.')[0])
        if major < 2:
            print(f"Warning: elan version {version} is outdated.")
            print("Please run: elan self update")
            return False
    except (IndexError, ValueError):
        print(f"Warning: Couldn't parse elan version: {stdout}")
        return False

    return True

def create_project(work_dir: LeanWorkDir) -> bool:
    """Create the project using mathlib4 toolchain if it doesn't exist."""
    if work_dir.project_dir.exists():
        print(f"Project directory {work_dir.project_dir} already exists, skipping creation...")
        return True

    # Use mathlib4's toolchain to create the project
    return run_command([
        'lake',
        '+leanprover-community/mathlib4:lean-toolchain',
        'new',
        str(work_dir.project_name)
    ], work_dir.base_dir)[0]

def setup_repl(work_dir: LeanWorkDir) -> bool:
    """Clone and build the REPL repository if not already built."""
    # Check if REPL is already built
    if work_dir.repl_binary.exists():
        print("REPL already built, skipping setup...")
        return True

    # Create test directory if it doesn't exist
    work_dir.test_dir.mkdir(parents=True, exist_ok=True)

    # Clone REPL repository if not exists
    if not work_dir.repl_dir.exists():
        success, _, _ = run_command([
            'git', 'clone',
            'https://github.com/leanprover-community/repl.git',
            str(work_dir.repl_dir)
        ])
        if not success:
            return False

    # Build REPL
    return run_command(['lake', 'build'], work_dir.repl_dir)[0]

def run_lean_file(work_dir: LeanWorkDir, content: str) -> tuple[bool, Any]:
    """Run a Lean file through the REPL."""
    if not work_dir.repl_binary.exists():
        print("Error: REPL binary not found!")
        return False, None

    # Create a temporary file in the project directory
    try:
        with tempfile.NamedTemporaryFile(mode='w',
                                       suffix='.lean',
                                       dir=work_dir.project_dir,
                                       delete=False) as tf:
            tf.write(content)
            temp_path = tf.name

        # Run the file through REPL
        cmd_json = json.dumps({"path": temp_path, "allTactics": True})
        success, stdout, stderr = run_command(
            ['lake', 'env', str(work_dir.repl_binary)],
            work_dir.project_dir,
            stdin_input=cmd_json + "\n",
            silent=True
        )

        # Clean up temp file
        Path(temp_path).unlink()

        if not success:
            print(f"REPL execution failed: {stderr}")
            return False, None

        try:
            # Take the last non-empty line as the result
            json_lines = [line for line in stdout.splitlines() if line.strip()]
            if not json_lines:
                print("No output from REPL")
                return False, None
            result = json.loads(json_lines[-1])
            return True, result
        except json.JSONDecodeError as e:
            print(f"Error parsing REPL output: {e}")
            print(f"Raw output: {stdout}")
            return False, None

    except Exception as e:
        print(f"Error running Lean file: {e}")
        return False, None

def setup(base_dir: str | None = None, project_name: str = "lean_project") -> Optional[LeanWorkDir]:
    """Run the complete setup process and return the work directory info."""
    work_dir = LeanWorkDir.create(project_name, base_dir)

    steps = [
        (check_elan_version, "Checking elan version", []),
        (create_project, "Creating/Loading project", [work_dir]),
        (setup_repl, "Setting up REPL", [work_dir]),
    ]

    for step_func, step_desc, args in steps:
        print(f"\n=== {step_desc} ===")
        if not step_func(*args):
            print(f"Failed at: {step_desc}")
            return None

    # Test REPL with a basic file
    print("\n=== Testing REPL ===")
    test_file_content = """
import Mathlib.Tactic

example : 1 + 1 = 2 := by rfl
"""

    print("\nTesting with file:")
    print(test_file_content)
    success, result = run_lean_file(work_dir, test_file_content)
    if not success:
        print("Warning: REPL test failed")
    else:
        print(f"Result: {json.dumps(result, indent=2)}")

    print(f"\nSetup completed successfully!")
    print(f"Project directory: {work_dir.project_dir}")
    print(f"REPL binary: {work_dir.repl_binary}")

    return work_dir

def main():
    parser = argparse.ArgumentParser(
        description='Setup a Lean project with mathlib4 and REPL support'
    )
    parser.add_argument('project_name', help='Name of the project to create')
    parser.add_argument('--dir', help='Base directory for the project')

    args = parser.parse_args()

    work_dir = setup(args.dir, args.project_name)
    if work_dir is None:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()