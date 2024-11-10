import subprocess
import json
import tempfile
import traceback
from typing import Dict

def verify(
    code: str,
    lean_workspace: str,
    timeout: int = 300
) -> Dict:
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

            # Check if there are any sorries in the proof
            has_sorry = len(result.get('sorries', [])) > 0

            is_error = any(msg['severity'] == 'error' for msg in result.get('messages', []))

            return {
                'pass': (not has_sorry) and (not is_error),
                'messages': result.get('messages', []),
                'result': result,
                'error': None
            }

    except subprocess.TimeoutExpired:
        return {
            "pass": False,
            "error": f"Verification timed out after {timeout} seconds",
            "traceback": traceback.format_exc()
        }
    except Exception as e:
        return {
            'pass': False,
            'messages': [{'severity': 'error', 'data': str(e)}],
            'result': None,
            'error': str(e)
        }

if __name__ == "__main__":
    # Test example
    test_code = """import Mathlib
import Aesop
import Mathlib.Tactic

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
  (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by {
  simp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero, Nat.succ_add]

  have h₁' : a * r = 2 := by {
    simpa [h₀] using h₁
  }

  have h₂' : a * r ^ 3 = 6 := by {
    simpa [h₀] using h₂
  }

  have h₃ : r ^ 2 = 3 := by {
    nlinarith
  }

  have h₄ : a = 2 / Real.sqrt 3 ∨ a = -(2 / Real.sqrt 3) := by {
    apply eq_or_eq_neg_of_sq_eq_sq
    field_simp
    nlinarith
  }

  simpa [h₀] using h₄
}"""

    result = verify(
        code=test_code,
        lean_workspace="./mathlib4",
        timeout=300
    )

    # Print results
    print(f"Verification result: {'✓' if result['pass'] else '✗'}")

    print(json.dumps(result['result'], indent=2))

    if result.get('error'):
        print("\nError:", result['error'])
    else:
        print("\nMessages:")
        for msg in result['messages']:
            if msg['severity'] == 'error':
                print(f"Error at line {msg['pos']['line']}: {msg['data']}")
            elif msg['severity'] == 'info':
                print(f"Info: {msg['data']}")