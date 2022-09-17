def check_cmake_txt(dir_: str, line_num: int, line_mod: str) -> bool:
    with open(dir_, 'r', encoding="utf-8") as file:
        data = file.readlines()
    
    return data[line_num] == line_mod


def modify_cmake_txt(dir_: str, line_num: int, line_mod: str) -> None:
    with open(dir_, 'r', encoding="utf-8") as file:
        data = file.readlines()
    
    data[line_num] = line_mod
        
    with open(dir_, 'w', encoding="utf-8") as file:
        file.writelines(data)