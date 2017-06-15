# data-Mining-Proj-2
zqwerty & llylly

#### preprocess

+ `src/preprocess.py:readXML(dir, dest)`

  读取 xml 返回列表 docs:[(doc-id,category,full_text),...]

  docs 使用 pickle 保存在 data_slice 目录下，读取可使用 `pickle.load(open('../data_slice/data.p'))`

  data.p 较大 -> gitignore

  **修订咗D，而家佢叫作readXMLs(dir, dest)**

  这货确实有D大，咁所以把每一个放到单独嘅文件里，默认係在data_pickle里->gitignore

  ~30s for 50000 passages

+ `src/preprocess.py:printXMLs(path, num)`

  从data_pickle里随机捻num个印出嚟

+ `src/preprocess.py:extractCategories(path, dest, freqThreshold)`

  由data_pickle搵出所有类别同埋佢哋嘅出现频数，超过threshold嘅分类单提出嚟，留俾分类嗰阵用，存进dest里

  ~ 25s for 50000 passages

+ `src/preprocess.py:extractWordBagVector(path, dest)`

  提出data_pickle嘅每篇文嘅有用词同佢哋嘅出现频数，咁把每篇文转为[{word: frequency}]嘅表示，存进dest里

  呢个函数有用enchant同NLTK呢两个NLP库，咁所以run前需要D时间（几个字到几个钟啫）下载同埋配置

  同时，有用词嘅提取也需要D时间，请谨慎调用

+ `src/preprocess.py:printWordBags(path, num)`

  从data_bagvec中随机捻num个出嚟，印出佢哋嘅word bag表示

  ~ 10099s for 50000 passages

+ `src/preprocess.py:getWordDict(path, dest, threshold)`

  从data_pickle中统计每个词出现嘅总频数同文章数，将总频数>=threshold嘅词加入词典里，再把佢存到dest嘅wordList.p里，形式为[[word, # of appeared docs]]

+ `src/preprocess.py:genTfIdfVecs(bagPath, dictPath, savePath, fold)`

  综合以上呢D信息，生成Tf-Idf表示，生成出嚟嘅表示嘅存储格式见函数嗰度嘅注释


  呢个函数嘅时间空间需求都较大，谨慎运行喇

  ~ 588s & 10G for 50000 passages