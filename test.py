#Author:Mingzhu W
import os
directory = "F:\\书籍文档"
def get_all_files(directory:str,files:list)->list:
    first_layer = os.listdir(directory)
    if "编程" in first_layer and "机器学习框架" in first_layer and "深度学习搜索引擎开发" in first_layer and "网络下载" in first_layer:
        first_layer.remove("机器学习框架")
        first_layer.remove("深度学习搜索引擎开发")
        first_layer.remove("编程")
        first_layer.remove("网络下载")
    elif "剑指offer第二版(PDF+源码)" in first_layer:
        first_layer.remove("剑指offer第二版(PDF+源码)")
    elif "内蒙古大学-泛函分析（国家级精品课）" in first_layer:
        first_layer.remove("内蒙古大学-泛函分析（国家级精品课）")
    elif "linear algebra" in first_layer:
        first_layer.remove("linear algebra")
    for ele in first_layer:
        entire_path = os.path.join(directory, ele)
        if not os.path.isdir(entire_path):
            if entire_path.endswith(".pdf") or entire_path.endswith(".txt"):
                files.append(ele)
        else:
            files.append("# "+ele)
            files = get_all_files(entire_path,files)
    return files
"""md格式文件中n个#加上空格表示n级标题"""
with open("books.md","w",encoding="utf-8") as f:
    books = get_all_files(directory,[])
    for book in books:
        f.write(book+"\n")

