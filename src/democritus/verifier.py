import subprocess
import json
import tempfile
import time
import traceback
from typing import Dict, Optional

def verify_lean_code(
    code: str,
    lean_workspace: str,
    timeout: int = 300
) -> dict:
    """
    Run Lean code and get goal states at each step.
    Assumes working directory is mathlib4 root.

    Args:
        code: Complete Lean code including imports and namespace
        lean_workspace: Path to mathlib4 workspace
        timeout: Maximum execution time in seconds

    Returns:
        Dict containing verification results
    """
    try:
        # Prepare command for Lean REPL
        command = {
            "cmd": code,
            "allTactics": True
        }
        message_str = json.dumps(command, ensure_ascii=False)

        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)

            outputs = subprocess.run(
                ["lake", "exe", "repl"],
                stdin=temp_file,
                capture_output=True,
                text=True,
                cwd=lean_workspace,
                timeout=timeout
            )

            if outputs.stderr:
                print("STDERR:", outputs.stderr)

            result = json.loads(outputs.stdout)
            return {
                "pass": not any(m['severity'] == 'error' for m in result.get('messages', [])),
                "messages": result.get('messages', []),
                "stdout": outputs.stdout,
                "stderr": outputs.stderr
            }

    except subprocess.TimeoutExpired:
        return {
            "pass": False,
            "error": f"Verification timed out after {timeout} seconds",
            "traceback": traceback.format_exc()
        }
    except Exception as e:
        return {
            "pass": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def verify_problem_json(
    problem: Dict,
    lean_workspace: str,
    timeout: int = 300
) -> Dict:
    """
    Verify a Lean theorem from a problem JSON object.

    Args:
        problem: Dict containing problem data with 'header' and 'formal_statement'
        lean_workspace: Path to mathlib4 workspace
        timeout: Maximum execution time in seconds

    Returns:
        Dict containing verification results
    """
    # Combine header and formal statement
    code_parts = []
    if 'header' in problem:
        code_parts.append(problem['header'])
    if 'formal_statement' in problem:
        code_parts.append(problem['formal_statement'])

    code = '\n'.join(code_parts)

    # Run verification
    result = verify_lean_code(code, lean_workspace, timeout)

    return {
        "problem_id": problem.get('id', 'unknown'),
        "verification_result": result
    }

if __name__ == "__main__":
    # Test example
    test_problem = {
        "id": "amc12a_2015_10",
        "header": """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
  (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
    sorry
"""
    }

    result = verify_problem_json(
        test_problem,
        lean_workspace="./mathlib4",  # Replace with your mathlib4 path
        timeout=300
    )

    print(result)

    # Print results
    print(f"Verification for {result['problem_id']}: {'✓' if result['verification_result']['pass'] else '✗'}")

    if result['verification_result'].get('error'):
        print("\nError:", result['verification_result']['error'])
    else:
        print("\nMessages:")
        for msg in result['verification_result']['messages']:
            if msg['severity'] == 'error':
                print(f"Error at line {msg['pos']['line']}: {msg['data']}")
            elif msg['severity'] == 'info':
                print(f"Info: {msg['data']}")