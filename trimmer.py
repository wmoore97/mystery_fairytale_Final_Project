# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:13:16 2019

@author: Moore

This script was designed to trim the header and footer info out of each ebook in teh data set

"""

import glob

path = "."

# Replace "mystery" for whichever folder of txt files you want to proccess, and make sure trimmer.py is in the paretn directory
for filename in glob.iglob(path+"/mystery/*.txt"):
    #iterate through each file in the folder, opening it
    with open(filename, encoding="utf8") as f:
        lines = f.read()
        #look for end of header
        start = lines.find("*** START OF THIS PROJECT GUTENBERG")
        #checks if end of header was found
        if start != -1:
            #search for start of footer
            end = lines.find("*** END OF THIS PROJECT GUTENBERG EBOOK")
            #text between header and footer
            trimmedText = lines[start :end]
            #close file
            f.close()
            #open file again, thihs time for reading 
            o = open(filename, "w", encoding = "utf8")
            #overwrite file with trimmed text
            o.write(trimmedText)
            #close file
            o.close()
        else:
            #print out filenames that weren't able to proccess correctly foe manual trimming
            print(filename)
            #closing of file
            f.close()        