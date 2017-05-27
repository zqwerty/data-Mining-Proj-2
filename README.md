# data-Mining-Proj-2
Data-Mining-Project-2

#### preprocess

`src/preprocess.py:readXML(dir)`

读取 xml 返回列表 docs:[(doc-id,category,full_text),...]

docs 使用 pickle 保存在 data_slice 目录下，读取可使用 `pickle.load(open('../data_slice/data.p'))`

data.p 较大 -> gitignore