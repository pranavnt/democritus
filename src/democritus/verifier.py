import subprocess
import json
import tempfile
import time
import traceback

def verify_lean_file(
    code: str,
    lake_path: str = "lake", # Path to lake executable
    lean_workspace: str = ".", # Path to your Lean project directory
    timeout: int = 300
) -> dict:
    """
    Verify a Lean 4 file and return the results.

    Args:
        code: The Lean code to verify
        lake_path: Path to the lake executable
        lean_workspace: Path to the Lean project directory (with lakefile.lean)
        timeout: Maximum time in seconds to wait for verification

    Returns:
        dict containing verification results including:
        - pass: Whether verification succeeded
        - complete: Whether proof is complete (no sorries)
        - errors: List of error messages
        - warnings: List of warning messages
        - verify_time: Time taken for verification
    """
    # Prepare the command to send to Lean
    command = {"cmd": code}
    message_str = json.dumps(command, ensure_ascii=False)

    start_time = time.time()
    try:
        # Create temporary file to hold the input
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)

            # Run Lean through lake
            outputs = subprocess.run(
                [lake_path, "exe", "repl"],
                stdin=temp_file,
                capture_output=True,
                text=True,
                cwd=lean_workspace,
                timeout=timeout
            )

            # Parse the output
            result = json.loads(outputs.stdout)

            # Format the response
            formatted_result = {
                "sorries": result.get('sorries', []),
                "errors": [m for m in result.get('messages', []) if m['severity'] == 'error'],
                "warnings": [m for m in result.get('messages', []) if m['severity'] == 'warning'],
                "infos": [m for m in result.get('messages', []) if m['severity'] == 'info'],
                "verified_code": code,
            }

            # Add pass/complete flags
            formatted_result['pass'] = not formatted_result['errors']
            formatted_result['complete'] = (
                formatted_result['pass'] and
                not formatted_result['sorries'] and
                not any("declaration uses 'sorry'" in warning['data']
                       for warning in formatted_result['warnings'])
            )

    except Exception as e:
        formatted_result = {
            "pass": False,
            "complete": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

    formatted_result['verify_time'] = time.time() - start_time
    return formatted_result

# Example usage
if __name__ == "__main__":
    # Example Lean code
    test_code = """
    theorem test_theorem : 2 + 2 = 4 := by
      simp
    """

    # Run verification
    result = verify_lean_file(
        test_code,
        lean_workspace="./mathlib4"  # Replace with your Lean project path
    )

    # Print results
    print(json.dumps(result, indent=2))