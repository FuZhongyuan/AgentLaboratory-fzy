import sqlite3

# 连接到数据库
conn = sqlite3.connect('papers.db')

# 创建一个游标对象
cursor = conn.cursor()

# 获取数据库中的所有表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# 遍历每个表并打印其信息
for table in tables:
    table_name = table[0]
    print(f"表名: {table_name}")

    # 获取表的列信息
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    print("列信息:")
    for column in columns:
        print(f"列名: {column[1]}, 类型: {column[2]}, 是否允许为空: {column[3]}, 默认值: {column[4]}, 是否主键: {column[5]}")
    print()

    # 获取表的数据
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    print("表数据:")
    for row in rows:
        print(row)
    print()

# 关闭连接
conn.close()