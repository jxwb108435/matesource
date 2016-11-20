import xlrd

book = xlrd.open_workbook('temp_01.xlsx')

print('表单数量: ', book.nsheets)
print('表单名称: ', book.sheet_names())

# 获取第1个表单
sh = book.sheet_by_index(0)

print("表单 %s 共 %d 行 %d 列" % (sh.name, sh.nrows, sh.ncols))

print("第1行第1列:", sh.cell_value(0, 0))
print("第1行第2列:", sh.cell_value(0, 1))

# 2 道路标识
print("第2行第2列_道路标识:", sh.cell_value(1, 1))

# 3 交通标识
print("第2行第3列_交通标识:", sh.cell_value(1, 2))

# 4 雨水标识
print("第2行第4列_雨水标识:", sh.cell_value(1, 3))

# 5 污水标识
print("第2行第5列_污水标识:", sh.cell_value(1, 4))

# 6 给水标识
print("第2行第5列_给水标识:", sh.cell_value(1, 5))

# 7 照明标识
print("第2行第5列_照明标识:", sh.cell_value(1, 6))

# 8 管线标识
print("第2行第5列_管线标识:", sh.cell_value(1, 7))

# 9 涵洞标识
print("第2行第5列_涵洞标识:", sh.cell_value(1, 8))

# 10 绿化标识
print("第2行第5列_绿化标识:", sh.cell_value(1, 9))

# 11 桥梁标识
print("第2行第5列_桥梁标识:", sh.cell_value(1, 10))

# 12 建筑标识
print("第2行第5列_建筑标识:", sh.cell_value(1, 11))
