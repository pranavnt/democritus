import subprocess
import json
import tempfile
import time
import traceback

def verify_lean_with_goals(
    code: str,
    lean_workspace: str,
    timeout: int = 300
) -> dict:
    """
    Run Lean code and get goal states at each step.
    Assumes working directory is mathlib4 root.
    """
    # Wrap the code in a namespace and add imports
    modified_lines = [
        "import Mathlib.Tactic",
        "namespace TestTheorem"
    ]

    # Add the original code
    modified_lines.extend(code.strip().split('\n'))

    # Close namespace
    modified_lines.append("end TestTheorem")

    # Join all lines
    modified_code = "\n".join(modified_lines)

    # Prepare command for Lean REPL
    command = {
        "cmd": modified_code,
        "allTactics": True
    }
    message_str = json.dumps(command, ensure_ascii=False)

    try:
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

    except Exception as e:
        return {
            "pass": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    # Test theorem
    test_code = """
theorem add_comm (n m : Nat) : n + m = m + n := by
  induction n with
  | zero =>
    simp
  | succ n ih =>
    simp [Nat.succ_add]
    rw [ih]
    rfl
"""

    # Run verification
    result = verify_lean_with_goals(
        test_code,
        lean_workspace="./mathlib4"  # Replace with your mathlib4 path
    )

    print(result)

    # Print results
    print("Verification:", "✓" if result['pass'] else "✗")

    if result.get('error'):
        print("\nError:", result['error'])
    else:
        print("\nMessages:")
        for msg in result['messages']:
            if msg['severity'] == 'error':
                print(f"Error at line {msg['pos']['line']}: {msg['data']}")
            elif msg['severity'] == 'info':
                print(f"Info: {msg['data']}")