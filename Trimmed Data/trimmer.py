# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:13:16 2019

@author: Moore
"""

import glob

path = "."

for filename in glob.iglob(path+"/mystery/*.txt"):
    with open(filename, encoding="utf8") as f:
        lines = f.read()
        start = lines.find("*** START OF THIS PROJECT GUTENBERG")
        if start != -1:
            end = lines.find("*** END OF THIS PROJECT GUTENBERG EBOOK")
            trimmedText = lines[start :end]
            f.close()
            o = open(filename, "w", encoding = "utf8")
            o.write(trimmedText)
            o.close()
        else:
            print(filename)
            f.close()        