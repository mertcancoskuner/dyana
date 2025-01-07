import sys

print("Hello World!")
print("This is going to stderr!", file=sys.stderr)
sys.exit(42)
