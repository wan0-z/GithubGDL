import json
import numpy as np
from tqdm import tqdm
from pathlib import Path


#%%
import os
print("工作目录:", os.getcwd())
#%%
graph_path = Path("../data/graphs")

face_attr = []
edge_attr = []

graph_name_list = list(graph_path.glob("*_result.json"))
print(f"Total number of graph files: {len(graph_name_list)}")


#%%
for graph_name in tqdm(graph_name_list, total=len(graph_name_list)):
    try:
        with open(graph_name, "r") as fp:
            data = json.load(fp)
        fp.close()
        attribute_map = data[1]
        face_attr.extend(attribute_map['graph_face_attr'])
        edge_attr.extend(attribute_map['graph_edge_attr'])
    except Exception as e:
        print(e)
        print(graph_name)
        continue

face_attr = np.array(face_attr)
edge_attr = np.array(edge_attr)

mean_face_attr = np.mean(face_attr, axis=0).tolist()
std_face_attr = np.std(face_attr, axis=0).tolist()
mean_edge_attr = np.mean(edge_attr, axis=0).tolist()
std_edge_attr = np.std(edge_attr, axis=0).tolist()

result_json = {
    "mean_face_attr": mean_face_attr,
    "std_face_attr": std_face_attr,
    "mean_edge_attr": mean_edge_attr,
    "std_edge_attr": std_edge_attr
}

json_file = open("../data/attr_stat.json", mode='w')
json.dump(result_json, json_file)

#%%
import json

graph_name = graph_name_list[0]  # 取第一个文件试试
print(graph_name)

with open(graph_name, "r") as fp:
    data = json.load(fp)

# 打印data的类型
print(f"data的类型是: {type(data)}")

print(f"data是列表，长度为: {len(data)}")
    # 打印前几个元素的类型
for i, item in enumerate(data[:5]):
    print(f"第{i}个元素的类型是: {type(item)}")

dict_data = data[1]

print(f"dict_data的类型是: {type(dict_data)}")
print(f"dict_data包含的key有: {list(dict_data.keys())}")

# 如果你想看每个key对应的值的类型，可以这样：
for key, value in dict_data.items():
    print(f"key: {key}, value类型: {type(value)}")

#%%
# data是列表，取第二个元素，是我们想要的字典

import json
import networkx as nx
import matplotlib.pyplot as plt
dict_data = data[1]

# 取graph部分
graph_data = dict_data['graph']

edges = graph_data['edges']
num_nodes = graph_data['num_nodes']

# 将两条列表合成边元组列表
edge_list = list(zip(edges[0], edges[1]))

# 建图
G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edge_list)

# 画图
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
plt.title('Graph Visualization')
# 保存图像到指定路径
plt.savefig('graph.png', dpi=300)  # dpi越高图像越清晰

plt.close()  # 关闭绘图窗口，释放内存

