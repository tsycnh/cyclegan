import os
import shutil
def get_dir_filelist_by_extension(dir, ext,with_parent_path=False):
    r = os.listdir(dir)
    file_list = []
    for item in r:
        if item.split('.')[-1] == ext:
            if with_parent_path:
                file_list.append(dir + '/' + item)
            else:
                file_list.append(item)
    return file_list

def create_new_empty_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def fixed_length(num,length):
    a = str(num)
    while len(a) < length:
        a = '0'+a
    return a

if __name__ == '__main__':
    one = 9
    b = fixed_length(9,5)
    print(b)