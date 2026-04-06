"""Write test results to file."""
import sys
sys.path.insert(0, '.')
from io import StringIO

# Capture output
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

# Run the tests
exec(open('run_8_tests.py').read())

# Get output
output = mystdout.getvalue()
sys.stdout = old_stdout

# Write to file
with open('test_output.md', 'w', encoding='utf-8') as f:
    f.write("# 8 Test Cases: RSA vs L0 Comparison\n\n")
    f.write("```\n")
    f.write(output)
    f.write("```\n")

print("Results saved to test_output.md")
