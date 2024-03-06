# hpyerspectral_classification
My final year project document titled Hyperspectral Classification based on Graph Convolutional Neural Networks. The main content is the reproduction of the results of some papers.

## structure
- config
    data_info.yaml 数据集路径
    yamls.py yaml文件读写
- data
    load_data.py 加载数据 主要是DataLoader类
- dataset
    三个数据集
- model
    segnet.py
    mdgcnnet.py  两个模型文件
- process 
    data_processor.py
    dimension_reduction.py
    graph_calculator.py
    neighbor.py 数据处理过程的参数
- run 
    计划放置可复用的训练、验证、测试代码
- util
    config.py 从命令行读取超参数，设置随机种子
    misc.py 一些优化器、device设置创建文件夹等函数
    metrics.py 评估模型性能的函数
    pixel_predtion.py 一个特殊的用于mdgcn输出结果处理的函数
    show.py 数据可视化类

DMSGcnRun.py 从run中特化的DMSgcn训练过程
MDGcnRun.py 从run中特化的MDGCN训练过程
main.py 训练入口，可选择的训练模型