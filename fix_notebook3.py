
import re

path = r"C:\Users\Khalil\Desktop\Deep Learning project\consignes_projet_final_classification_images.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the broken code cell 25
# It starts with '"id": "P_tUVvvxxR3F"' and has broken raw python after outputs

start_marker = '"id": "P_tUVvvxxR3F"'
start_idx = content.find(start_marker)
if start_idx == -1:
    print("Marker not found")
    exit(1)

# Find the start of the cell object (go backwards to find '{')
cell_start = content.rfind('{', 0, start_idx)

# Find the end of the cell - look for the next '},' followed by whitespace and then a new cell or end of cells array
# We'll search forward from the broken cell for the pattern that starts the next cell
# The next cell starts with whitespace + '{' followed by '"cell_type": "markdown"'
# Actually, looking at line 1156 of the file:  },
#   {
#    "cell_type": "code",

# Let's find the next occurrence of '"cell_type":' that would be the next cell
# But we need to be careful - there might be other "cell_type" inside strings.
# A safer approach: find the closing '}' of the current broken cell.
# We know the next cell starts with '  {\n   "cell_type": "markdown"'

# Find the position of the next markdown cell after this code cell
# Search for the next cell_type marker that follows the current position
rest = content[start_idx:]
# Find the end of the cell: search for the next '},' that closes a cell
# The cell ends with '  },' (line 1150 of current file) 
# Let's find the matching '},' after the broken content

# Actually, let's find the next cell start by looking for '  {\n   "cell_type": "markdown"' or similar
next_cell_pattern = '  {\n   "cell_type": "markdown"'
next_cell_idx = content.find(next_cell_pattern, start_idx)
if next_cell_idx == -1:
    print("Next cell not found with expected pattern")
    exit(1)

# The cell we want to replace ends just before next_cell_idx
# Let's go back from next_cell_idx to find '},\n' which is the end of the previous cell
end_idx = content.rfind('  },\n', start_idx, next_cell_idx)
if end_idx == -1:
    # Try without newline
    end_idx = content.rfind('  },', start_idx, next_cell_idx)
    
end_idx += len('  },')

print(f"Replacing block from index {cell_start} to {end_idx}")
print("Old block preview:")
print(content[cell_start:cell_start+300])
print("...")
print("Next cell preview:")
print(content[next_cell_idx:next_cell_idx+200])
