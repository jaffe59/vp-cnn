from __future__ import print_function
import sys

with open(sys.argv[1], "r") as iff:
    for line in iff:
        fields = line.strip().split(" ")
        result = fields[0] + "\t" + " ".join(fields[1:])
        print(result)
