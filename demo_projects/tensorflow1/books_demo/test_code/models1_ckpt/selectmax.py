"""
@Time : 2021/4/14 15:31
@Author : wmingzhu
@Annotation : 
"""
import os



def find_latest(parent_path:str)->str:
    file_list = os.listdir(parent_path)
    count = 0
    max = 0
    meta_file = ""
    for file in file_list:
        if file.endswith(".meta"):
            count += 1

    if count == 1:
        for file in file_list:
            if file.endswith(".meta"):
                meta_file = parent_path + file if parent_path.endswith("/") else parent_path +"/" + file
                break
    else:
        for file in file_list:
            if file.endswith(".meta"):
                number = int(os.path.splitext(file)[0][-1])
                max = number if max < number else max

        for file in file_list:
            if file.endswith(".meta"):
                if int(os.path.splitext(file)[0][-1]) == max:
                    meta_file = parent_path + file if parent_path.endswith("/") else parent_path + "/" + file
                    break

    return meta_file

print("最终拿到的最新的meta文件是:",find_latest("./"))
