import os 
import re
import clipboard

# Preprocesses matlab code in order to rewrite it as python

in_fpath = "youbot/focused/plot_matrix.m"
use_clipboard = False   # If true, ignores the filepath and takes input from the clipboard instead

if use_clipboard:
    source = clipboard.paste().replace("\r\n", "\n").split("\n")
else:
    out_fpath = in_fpath.replace(".m", ".py")        
    if os.path.exists(out_fpath):
        input("Warning: file exists. Press any key to overwrite.")
    source = [line.rstrip() for line in open(in_fpath, "r")]

def group_sub(group, repl):
    """
    Returns a function operating on an sre_match object that substitutes a given group in the 
    match with a given string or with the return value of a function operating on the contents of 
    the group.
      
    :param group: the index of the capturing group.
    :param repl: either a string or a function that takes a string and returns a string.
    :return: the string corresponding to the match with the group substring substituted by its 
    replacement.
    """
    def wrapper(sre_match):
        if isinstance(repl, str):
            group_string = repl
        else:
            group_string = repl(sre_match)
        full_string = sre_match.group(0)
        prefix = full_string[:sre_match.start(group) - sre_match.start(0)]
        suffix = full_string[sre_match.end(group) - sre_match.start(0):]
        return prefix + group_string + suffix
    return wrapper

# Generic matlab code operations
operations = [
    (r'%', lambda m: m.group(0).replace('%', '#')), # Comment character: % -> #
    (r'^\s*end\s*', ''),            # Removal of end statement for if, else, for 
    (r'(;)\s*', group_sub(1, '')),  # Removal of trailing semicolons 
    (r'(\W|^)(true)(\W|$)', group_sub(2, 'True')),      # True statement: true -> True
    (r'(\W|^)(false)(\W|$)', group_sub(2, 'False')),    # False statement: false -> False
    (r'(\W|^)(elseif)(\W|$)', group_sub(2, 'elif')),    # Else if statement: elseif -> elif
    (r'(\W|^)(disp|fprintf)\(.*\)', group_sub(2, 'print')),     # Print statements
    (r'^\s*(function|if|for|while|else|elif)( |$).*()\s*', group_sub(3, ':')),  # Semicolons
    (r'\s*function ', 'def '),      # Function definition: function -> def
    (r'^\s*for .* ?(= ?.+:(.+)):', group_sub(1, lambda m: "in range(%s)" % m.group(2))),    # For
]
## Operations specific to the VREP API
# operations = [
#     (r'def .*\((clientID)', group_sub(1, 'self')),
#     (r'\S*\(()clientID', group_sub(1, 'self.')),
#     (r'()^\s*def simx', group_sub(1, '    @validate_output\n')),
# ]
        
output = ""
for line in source:
    for operation in operations:
        line = re.sub(*operation, line)
    output += "%s\n" % line
    
print("Output:\n%s" % output)
if use_clipboard:
    clipboard.copy(output)
else:
    out_file = open(out_fpath, "w")
    out_file.write(output)
    

