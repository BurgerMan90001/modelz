"""_summary_
Main entry point
"""
import csv
from typing import Tuple

import numpy as np
from sklearn import tree

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def read_csv(url:str) -> Tuple[list[list[str]], list[str]]:
    """_summary_
    """
    with open(url, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        return iterate_reader(reader , 5)      
def slice_y(target_list: list[str], slice_index: int) -> Tuple[list[str],str]:
    """_summary_
    Args:
        target_list (list[str]): List to be sliced
        slice_index (_type_): The index to slice at in the target list
    Returns:
        Tuple[list[str],str]: The sliced list and y value
    """
    y = target_list[len(target_list)-1]
    return target_list[:slice_index], y
def slice_list_at_index(
     target_list: list[object],
     slice_at_index: int) -> Tuple[list[object], list[object]]:
    """_summary_
    slices a list into half based on the slice_at_index
    Args:
         list (list[object]): _description_
         slice_at_index (int): _description_
    Returns:
         Tuple[list[object], list[object]]: _description_
    """
    return target_list[:slice_at_index], target_list[-slice_at_index:]

def iterate_reader(reader, max_iterations: int) -> Tuple[list[list[str]], list[str]]:
    """ iterates through the read csv file 
    Args:
         reader (Reader): CSV reader
    """    
    y_list: list[str] = []
    x_list: list[list[str]] = []
    slice_index: int = 16
    for row in reader: # type: ignore
        #print(slice_list_at_index(row,3))
        y = row[len(row)-1]
        x = row[:slice_index]
        y_list.append(y)
        x_list.append(x)
        """""
        if count >= max_iterations:
             break
        count += 1
        """
    return x_list, y_list

def to_binary(string:str):
    """_summary_
    Returns:
        _type_: _description_
    """
    return ''.join(format(ord(char), '08b') for char in string)
data = read_csv('pythone/big_data/bank-full.csv')


le = LabelEncoder()


X = data[0]
Y = data[1]
test = ['apple', 'pear', 'apple', 'orange']
#from sklearn.preprocessing import LabelBinarizer
clf = tree.DecisionTreeClassifier()
print(X[0][0])


def encode_y_classes(y_list:list[str]):
    """ Encodes a list of y labels between 0 and n_classes-1.
    Args:
        y_list (list[str]): _description_
    """
    y_encoded = le.fit_transform(y_list)
    #test.
    #le.transform(y_list)
    print(y_encoded)
    return y_encoded
encode_y_classes(test)

#model = SVC()
#Y_dense = LabelBinarizer().fit_transform(Y)
#print(Y_dense)
#clf.fit(X, Y)

#prediction = clf.predict([[""]])

#print(prediction)
