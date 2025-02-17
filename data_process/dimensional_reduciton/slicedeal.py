# -*- coding: utf-8 -*-
# 将根目录加入sys.path中,解决命令行找不到包的问题
import re
import sys
import shutil

import pandas as pd
import regex

from datapro import get_curr_data, getnewcolumns, cal_column

rootPath = '/home/youhuazi/Context_CC'
sys.path.append(rootPath)
import os
from os import listdir

# slice_path = r'/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/'
# name = []
# for filepath, dirnames, filenames in os.walk(r'' + str(slice_path)):
#     for filename in filenames:
#         name.append(str(filepath))
#         break
# name.sort()
# print(name)
"""
#Delete the files ending with ~ which are wrong
for spath in name:
    for file_name in listdir(spath):
        if file_name.endswith('~'):
            path=os.path.join(spath,file_name)
            #print(path)
            os.remove(path)
"""

"""
#对以_结尾的数据文件的重命名，方便统一处理
for spath in name:
    for file_name in listdir(spath):
        path = os.path.join(spath, file_name)
        if path.endswith('_'):
            print(path)
            print(path[:-1])
            os.rename(path,path[:-1])
"""
"""
#将没有标明是report还是root的数据文件重命名
for spath in name:
    for file_name in listdir(spath):
        path = os.path.join(spath, file_name)
        if not path.endswith(")"):
        #if not path.endswith('(root)') and not path.endswith('(report)'):
            #os.rename(path, path+"(report)")
            print(path)
"""

"""
#这些文件判断为空
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/lang/54_/LocaleUtils(root)
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/lang/13=/SerializationUtils268(root=report)
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/lang/55_/StopWatch118(root)
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/lang/55_/StopWatch119(root)
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/math/70_/72(root)
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/math/98_/BigMatrixImpl991(rpoot)
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/math/98_/779(root)
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/math/79/KMeansPlusPlusCluster91(report)
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/math/79/MathUtils1624(root)
#并对这些文件进行删除
for spath in name:
    for file_name in listdir(spath):
        path = os.path.join(spath, file_name)
        if os.path.getsize(path)==0:
            print(path)
            os.remove(path);
"""
"""
#用于判断是否存在空的文件夹并删除，目前经过上述处理未发现空文件夹
# 存在一定的问题，手撕了几个文件夹
for spath in name:
    file_list=listdir(spath)
    if len(file_list)==0:
        print(spath)
        #os.rmdir(spath)
"""
"""
#以root结尾的数据是不能够使用的，因为具有这就是我们要找的错误语句，在实际情况中是得不到的数据
#但暂且将其保留，作为bugline，其中的数据不能使用
"""
"""
#重命名文件路径以与defects4j的数据路径相匹配
for spath in name:
    print(spath)
    mainpath=re.findall('(.*?)/\d.*?',spath)[0]
    number=re.findall(".*?/(\d+)",spath)[0]
    newpath=os.path.join(mainpath,number)
    print(newpath)
    os.rename(spath,newpath)
"""
"""
#找到那些只有行号没有类名的数据文件，这些文件的类名与同文件夹下的其他有类名的数据相同
#因此，根据这个修改重命名这些文件
dic = {}
for spath in name:
    file_names = listdir(spath)
    file_names.sort()
    for file_name in file_names:
        if re.findall('^\d+.*?', file_name):
            # print(spath)
            # print(file_names)
            dic[spath] = file_names
            break
# print(dic)
for key, item in dic.items():
    #print(key)
    #print(item)
    class_name = re.findall(r"(\w*[a-zA-Z]+)\s?\d+.*?", item[-1])[0]
    #print(class_name)
    for file in item:
        if re.findall('^\d+.*', file):
            #print(file)
            new_name = class_name + file
            #print(new_name)
            oldpath=os.path.join(key,file)
            newpath=os.path.join(key,new_name)
            print(oldpath)
            print(newpath)
            os.rename(oldpath,newpath)
"""
"""
#去掉数据文件中的空格，方便接下去的处理
for spath in name:
    for file_name in listdir(spath):
        if " " in file_name:
            print(file_name)
            new_file_name=file_name.replace(" ",'')
            print(new_file_name)
            old_path=os.path.join(spath,file_name)
            new_path=os.path.join(spath,new_file_name)
            os.renames(old_path,new_path)
"""

"""
考虑到这个类不是很理解，暂时放弃这个数据
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Mockito/4
Reporter(root)


for spath in name:
    for file_name in listdir(spath):
        if not bool(re.search(r'\d',file_name)):
            print(spath)
            print(file_name)
"""

"""
#寻找命名中有test的文件，但是不清楚接下来如何处理，暂时搁置
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Chart/1
['Test409(report)', 'AbstractCategoryItemRenderer1797(root=report)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Chart/11
['Tests212(report)', 'ShapeUtilities275(root)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Chart/25
['test236(report)', 'test291(report)', 'test208(report)', 'StatisticalbarRenderer315(root)', 'StatisticalbarRenderer459(root)', 'test263(report)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Chart/7
['TimePeriodValues300(root)', 'Test377(report)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Time/17
['DateTimeZone.adjustOffset1174(root)', 'TestDateTimeZoneCutover.java_1259(report)', 'DateTimeZone.adjustOffset1169(root)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Time/21
['TestDateTimeZone698(report)', 'DefaultNameProvider72(root)', 'TestDateTimeZone666(report)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Time/23
['DateTimeZone586(root)', 'TestDateTimeZone282(report)']

Process finished with exit code 0

for spath in name:
    flag=False
    for file_name in listdir(spath):
        if 'est' in file_name:
            flag=True
            break
    if flag:
        print(spath)
        print(listdir(spath))
"""

"""

# 对上述文件进行了删除
for spath in name:
    file_names = listdir(spath)
    delete_flag=False
    for file_name in file_names:
        if "test" in file_name or "Test" in file_name:
            print(spath+"存在暂时无法处理的test的文件")
            delete_flag=True
            break
    if delete_flag:
        print(spath)
        shutil.rmtree(spath)
"""

"""
# 在Time这个数据集中，很多文件的命名都带上了最后的指令集中的操作，因此对这部分文件进行重命名
# 将多有的操作名称去掉
for spath in name:
    for file_name in listdir(spath):
        if bool(re.search(r'.*?\..*?',file_name)):
            new_file_name = re.findall(r'(.*?)\..*?(\d+.*)',file_name)
            new_file_name=new_file_name[0][0]+new_file_name[0][1]
            print(spath)
            print(file_name)
            print(new_file_name)
            old_path=os.path.join(spath,file_name)
            new_path=os.path.join(spath,new_file_name)
            os.rename(old_path,new_path)
"""

"""
# 以下数据没有report文件，进行数据删除

/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Chart/14
['XYPlot2293(root)', 'XYPlot2529(root)', 'CategoryPlot2448(root)', 'CategoryPlot2166(root)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Chart/6
['ShapeList111(root)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Time/10
['BaseSingleFieldPeriod104(root)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Time/16
['DateTimeFormatter708(root)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Time/19
['DateTimeZone900(root)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Time/3
['addYears660(root)', 'add639(root)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Time/4
['Partial464(root)']
/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Time/9
['DateTimeZone267(root)', 'DateTimeZone263(root)', 'DateTimeZone265(root)']

for spath in name:
    flag=False
    file_names=listdir(spath)
    for file_name in file_names:
        if "report" in file_name:
            flag=True
            break
    if not flag:
        print(spath)
        print(file_names)
        shutil.rmtree(spath)
"""
"""
# 删掉bugline中大于1000的无用的数据
bugline_path="/home/youhuazi/Context_CC/defects4j/test/buggy-lines"
file_names=listdir(bugline_path)
cnt=0
for file_name in file_names:
    if file_name.endswith(".candidates"): # .buggy.lines
        if bool(re.search('\d{4,9}',file_name)):
            print(file_name)
            os.remove(os.path.join(bugline_path,file_name))
            cnt+=1
print(cnt)
"""


def get_raw_data(spath) -> list:
    """
    spath: 原始数据的文件夹的绝对路径
    作用：
        返回切片数据所对应的原始数据的代码行
    """
    file_path=os.path.join(spath,"spectra")
    with open(file_path, 'r') as f:
        spectra_data = f.readlines()
    spectra_data = [i.replace('\n', '') for i in spectra_data]
    return spectra_data


def get_buggy_line( bugline_path) -> list:
    """
    bugline_path: 原始数据的错误代码文件夹的绝对路径，具体到文件
    作用：
        返回切片数据所对应的原始数据的错误代码行
    """
    with open(bugline_path, 'r',encoding='unicode_escape') as f:
        data = f.readlines()
        #print(data)
        bug_data = list()
        #print(len(data))
        for bugline in data:
            bugcode = re.findall(r"(.*).java(#\d+)#.*", bugline)
            #print(bugcode)
            bugcode = bugcode[0][0] + bugcode[0][1]
            bugcode = bugcode.replace('/', '.')
            bug_data.append(bugcode)
    bug_data.sort()
    #print(bug_data)
    return bug_data


def get_context(spath) -> list:
    """
    spath: spath: 切片数据的文件夹的绝对路径
    作用：
        返回文件夹下所有report文件构成的上下文，包括report文件名所代表的报错行
    """
    # 对数据文件中的代码行进行提取
    context = []
    file_names = listdir(spath)
    for file_name in file_names:
        each_context = []
        if "report" in file_name:
            # 将文件名代表的代码行提取出来
            error_line = re.findall(r'(.*?)\(', file_name)[0]
            error_line = re.findall(r'(.*?)(\d+$)', error_line)
            error_class = error_line[0][0]

            # 构造文件路径，打开对应文件进行处理
            file_path = os.path.join(spath, file_name)
            with open(file_path, 'r') as f:
                data = f.readlines()
                for line in data:
                    each_context.append(line.split(' ')[0])
            # 对代码行的格式进行修改
            new_each_context = []
            for i in each_context:
                new_row = re.findall(r'(.*)\..*?:(\d+)', i)
                new_code = new_row[0][0] + '#' + new_row[0][1]
                new_each_context.append(new_code)
            # 寻找文件名对应的代码行
            full_error_class = error_class
            # 对所有的代码行匹配
            for i in new_each_context:
                if bool(re.search(f'.*?\.{error_class}#.*?', i, re.I)):
                    # 忽略类名的大小写不同
                    full_error_class = re.findall(r'(.*?)#.*', i, re.I)[0]
                    break
            # 判断是否找到了完整的类名，并输出一定信息，方便调试

            if full_error_class == error_class:
                print("没有找到对应的完整类名")
                print("路径为：" + spath)
                print("文件名为：" + file_name)
                print("文件类名为：" + error_class)
                error_line = error_class + "#" + error_line[0][1]
            else:
                error_line = full_error_class + "#" + error_line[0][1]
            new_each_context.append(error_line)
            # 对数据去重
            new_each_context = list(set(new_each_context))
            context.extend(new_each_context)
    context = list(set(context))
    context.sort()
    return context


# 这时候应该是在原始数据中进行数据处理
slice_des_path = "/home/youhuazi/Context_CC/defects4j/ContextData/SliceData"
bugline_path = "/home/youhuazi/Context_CC/defects4j/test/buggy-lines"
raw_data_path = "/home/youhuazi/Context_CC/defects4j/test/Data"
slice_path = r'/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/'

name = []
for filepath, dirnames, filenames in os.walk(r'' + str(raw_data_path)):
    for filename in filenames:
        name.append(str(filepath))
        break
name=list(filter(lambda x: regex.findall(r'.*\d$', x) !=[], name))
print(name)
name.sort()


slice_data=list()
raw_data=list()
no_data=list()
# 先判断是否有切片数据
# 对没有切片结果的数据使用所有错误样例执行结果的并集作为切片结果
# 还是先判断切片结果中是否包含了错误语句，如不包含则舍弃

name=['/home/youhuazi/Context_CC/defects4j/test/Data/Chart/4']
for spath in name:
    # if 1:
    # spath="/home/youhuazi/Context_CC/defects4j/test/defects4j_slice/Chart/16"
    # 获取类名和编号
    class_name = re.findall(r".*/(.*?)/\d+", spath)[0]
    class_name_number = re.findall(r".*/(\d+)$", spath)[0]
    print("开始处理数据"+str(class_name)+"-"+str(class_name_number))
    # 构造切片数据路径
    slice_file_path=os.path.join(slice_path,class_name,class_name_number)
    # 构造错误语句路径
    bugline_name = class_name + '-' + class_name_number + ".buggy.lines"
    bug_line_path=os.path.join(bugline_path,bugline_name)
    # 构造切片结果的路径
    des_data_path = os.path.join(slice_des_path, os.path.join(class_name, class_name_number))
    # 获得原始数据集和类名映射的列表
    features, label, data, columns = get_curr_data(spath)
    # 获得错误代码行
    bug_data = get_buggy_line(bug_line_path)
    # 对bug_lines进行数值映射
    fault_line = [cal_column(i, columns) for i in bug_data]
    # 判断路径是否存在 即是否有上下文数据
    if os.path.exists(slice_file_path):
        # 获得原始代码行
        spectra_data = get_raw_data(spath)
        # 获得上下文
        context = get_context(slice_file_path)
        # 寻找上下文与原代码错误行的交集
        bug_lines = list(set(context) & set(bug_data))
        # 寻找在上下文中没有在源代码中的代码行
        spectra = list(set(context) - set(spectra_data))
        # 判断是否成功获得上下文 上下文中是否有源代码中不存在的代码行  上下文中是否有错误语句
        if len(context) > 0 and len(spectra) == 0 and len(bug_lines)>0:
            # 对上下文进行数值映射
            context_line = [cal_column(i, columns) for i in context]
            # print("所有的上下文都在原代码中")
            # 获取原代码的行号
            column = features.columns
            column = list(column)
            # # 将上下文按照原代码中的顺序排序
            ordered_context_line = []
            for i in column:
                if i in context_line:
                    ordered_context_line.append(i)
            # 在原始数据集中提取出上下文构成新的数据集
            context_feature_data = data[ordered_context_line]
            # print(label)
            # print(context_feature_data)
            context_data = pd.concat([context_feature_data, label], axis=1)
            # 判断路径是否存在
            if not os.path.exists(des_data_path):
                os.makedirs(des_data_path)
            coverage_path = os.path.join(des_data_path, "matrix.csv")
            bug_path = os.path.join(des_data_path, "bugline.txt")
            # if not os.path.exists(coverage_path):
            # 将覆盖矩阵写入
            context_data.to_csv(coverage_path, index=True, header=True)
            #print(coverage_path)
            #if not os.path.exists(bug_path):
            # 将错误代码写入
            with open(bug_path, 'w') as f:
                for i in fault_line:
                    f.write(str(i)+'\n')
            slice_data.append(spath)
        # 获取上下文失败  切片中存在原始数据中没有的代码行或切片中无错误行
        else:
            # 获得失败样例的集合
            context=features[data['error']==1]
            # 对执行情况求和
            count=context.sum(axis=0)
            # 求失败样例执行情况的并集作为新的上下文
            context_index = list(count.index[count >0])
            # 判断是否为空
            if len(context_index)!=0:
                # print(context_index)
                #print(fault_line)
                # 求bug_line与现上下文的交集
                bug_line=list(set(context_index)&set(fault_line))
                if len(bug_line)==0:
                    #print("没有共同错误语句")
                    no_data.append(spath)
                else:
                    # 在原始数据集中提取出上下文构成新的数据集
                    context_feature_data = data[context_index]
                    # print(label)
                    # print(context_feature_data)
                    context_data = pd.concat([context_feature_data, label], axis=1)
                    # 判断路径是否存在
                    if not os.path.exists(des_data_path):
                        os.makedirs(des_data_path)
                    coverage_path = os.path.join(des_data_path, "matrix.csv")
                    bug_path = os.path.join(des_data_path, "bugline.txt")
                    # if not os.path.exists(coverage_path):
                    # 将覆盖矩阵写入
                    context_data.to_csv(coverage_path, index=True, header=True)
                    # print(coverage_path)
                    # if not os.path.exists(bug_path):
                    # 将错误代码写入
                    with open(bug_path, 'w') as f:
                        for i in fault_line:
                            f.write(str(i) + '\n')
                    raw_data.append(spath)
            else:
                no_data.append(spath)
    else:
        # 获得失败样例的集合
        context = features[data['error'] == 1]
        # 对执行情况求和
        count = context.sum(axis=0)
        # print(count)
        # 求失败样例执行情况的并集作为新的上下文
        context_index = list(count.index[count > 0])
        # 判断是否为空
        if len(context_index) != 0:
            # 求bug_line与现上下文的交集
            # print(context_index)
            # print(fault_line)
            bug_line = list(set(context_index) & set(fault_line))
            if len(bug_line) == 0:
                #print("没有共同错误语句")
                no_data.append(spath)
            else:
                # 在原始数据集中提取出上下文构成新的数据集
                context_feature_data = data[context_index]
                context_data = pd.concat([context_feature_data, label], axis=1)
                # 判断路径是否存在
                if not os.path.exists(des_data_path):
                    os.makedirs(des_data_path)
                coverage_path = os.path.join(des_data_path, "matrix.csv")
                bug_path = os.path.join(des_data_path, "bugline.txt")
                # 将覆盖矩阵写入
                ## context_data.to_csv(coverage_path, index=True, header=True)
                # if not os.path.exists(bug_path):
                # 将错误代码写入
                # with open(bug_path, 'w') as f:
                #     for i in fault_line:
                #         f.write(str(i) + '\n')
                raw_data.append(spath)



#
# print("真实切片数据：")
# print(slice_data)
# print("错误执行语句替代的数据：")
# print(raw_data)
# print("没有切片的数据：")
# print(no_data)
# print("总数据个数为："+str(len(name)))
# print("有上下文的个数为："+str(len(slice_data)+len(raw_data))+"\t占比："+str((len(slice_data)+len(raw_data))/len(name)))
# print("真实的切片数据个数是："+str(len(slice_data))+"\t占比："+str(len(slice_data)/len(name)))
# print("使用错误执行语句替代的数据个数为："+str(len(raw_data))+"\t占比："+str(len(raw_data)/len(name)))
# print("没有上下文的数据个数为："+str(len(no_data))+"\t占比："+str(len(no_data)/len(name)))
# data_info=["总数据个数为："+str(len(name)),
#            "有上下文的个数为："+str(len(slice_data)+len(raw_data))+"\t占比："+str((len(slice_data)+len(raw_data))/len(name)),
#            "真实的切片数据个数是："+str(len(slice_data))+"\t占比："+str(len(slice_data)/len(name)),
#            "使用错误执行语句替代的数据个数为："+str(len(raw_data))+"\t占比："+str(len(raw_data)/len(name)),
#            "没有上下文的数据个数为："+str(len(no_data))+"\t占比："+str(len(no_data)/len(name))]
# #
# # class_name= re.findall(r".*/(.*?)/\d+", name[0])[0]
# # info_path=os.path.join(slice_des_path,class_name,"info")
#
# with open(info_path, 'w') as f:
#     for i in data_info:
#         f.write(str(i) + '\n')
# # 将对应的文件路径写入文件
# slice_info_path=os.path.join(slice_des_path,class_name,"slice_info")
# with open(slice_info_path, 'w') as f:
#     for i in slice_data:
#         f.write(str(i) + '\n')
# raw_info_path=os.path.join(slice_des_path,class_name,"raw_info")
# with open(raw_info_path, 'w') as f:
#     for i in raw_data:
#         f.write(str(i) + '\n')
# no_info_path=os.path.join(slice_des_path,class_name,"no_info")
# with open(no_info_path, 'w') as f:
#     for i in no_data:
#         f.write(str(i) + '\n')