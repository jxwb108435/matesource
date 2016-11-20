# coding=utf-8
# -*- coding:cp936 -*-
"""
PdfFileReader(open("C:/Users/lin/Desktop/t1.pdf", 'rb')).getNumPages()
"""
from PyPDF2 import PdfFileMerger

f1 = open("C:/Users/lin/Desktop/t1.pdf", 'rb')
f2 = open("C:/Users/lin/Desktop/t2.pdf", 'rb')

merger = PdfFileMerger()
merger.append(f1)
merger.append(f2)

output = open("C:/Users/lin/Desktop/t.pdf", "wb")
merger.write(output)
output.close()

