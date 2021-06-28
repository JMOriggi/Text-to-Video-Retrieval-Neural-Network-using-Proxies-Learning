import _init_paths
from datasets.charades import Charades

dataset = Charades(1) # 0 for train 1 for test
# get an iterator using iter()
print(len(dataset))
my_iter = iter(dataset)

# Extract all features
'''# iterate through it using next()
while True:
    try:
        el = next(my_iter)
    except StopIteration:
        break'''
        
# Analyze element in database 
el = next(my_iter) 

'''
print('visual')
print(len(el['visual_input']))
print(len(el['vis_mask']))
print(el['vis_mask'])
print('text')
print(len(el['word_vectors']))
print(len(el['txt_mask']))
print('map')
print(len(el['map_gt']))
print(len(el['map_gt'][0]))

print(el['vid_idx'])
print(dataset.distinct_vid[el['vid_idx']])
import sklearn.preprocessing
#T = ['3MSZA', 'AMT7R','HVFXT','QIT2W','O1LMX','SVIXG', '7JHW2']
T = [0,3,55]
T = sklearn.preprocessing.label_binarize(T, classes = range(0,len(dataset.distinct_vid)))
print(T)
'''
