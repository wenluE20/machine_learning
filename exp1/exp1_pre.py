import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class TreeNode:
    """决策树节点类"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # 特征名称
        self.threshold = threshold  # 特征取值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶节点的值
        self.is_leaf = value is not None  # 是否为叶节点
        self.children = {}  # 多叉树的子节点

class ID3DecisionTree:
    """ID3决策树实现"""
    def __init__(self, max_depth=None, min_samples_split=1):
        self.root = None
        self.max_depth = max_depth  # 最大深度（预剪枝）
        self.min_samples_split = min_samples_split  # 最小分裂样本数（预剪枝）
    
    def entropy(self, y):
        """计算信息熵"""
        if len(y) == 0:
            return 0
        # 计算每个类的概率
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        # 计算熵
        return -sum(p * np.log2(p) for p in probabilities)
    
    def information_gain(self, X, y, feature):
        """计算信息增益"""
        # 计算父节点的熵
        parent_entropy = self.entropy(y)
        
        # 获取特征的所有可能取值
        unique_values = X[feature].unique()
        child_entropy = 0
        
        # 计算每个取值的条件熵
        for value in unique_values:
            subset_indices = X[feature] == value
            subset_y = y[subset_indices]
            weight = len(subset_y) / len(y)
            child_entropy += weight * self.entropy(subset_y)
        
        # 信息增益 = 父节点熵 - 子节点条件熵
        return parent_entropy - child_entropy
    
    def best_feature_to_split(self, X, y):
        """选择信息增益最大的特征进行分裂"""
        best_gain = -1
        best_feature = None
        
        for feature in X.columns:
            gain = self.information_gain(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        return best_feature, best_gain
    
    def fit(self, X, y):
        """训练决策树"""
        self.root = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        """递归构建决策树"""
        # 如果所有样本属于同一类别
        if len(y.unique()) == 1:
            return TreeNode(value=y.iloc[0])
        
        # 如果没有特征可分或达到最大深度或样本数太少
        if len(X.columns) == 0 or (self.max_depth is not None and depth >= self.max_depth) or len(X) < self.min_samples_split:
            # 返回样本中最多的类别
            return TreeNode(value=Counter(y).most_common(1)[0][0])
        
        # 选择最佳分裂特征
        best_feature, best_gain = self.best_feature_to_split(X, y)
        
        # 如果信息增益为0，返回多数类别
        if best_gain == 0:
            return TreeNode(value=Counter(y).most_common(1)[0][0])
        
        # 创建节点
        node = TreeNode(feature=best_feature)
        
        # 对每个特征值递归构建子树
        for value in X[best_feature].unique():
            subset_X = X[X[best_feature] == value].drop(columns=[best_feature])
            subset_y = y[X[best_feature] == value]
            
            if len(subset_X) == 0:
                # 如果子集为空，返回多数类别
                node.children[value] = TreeNode(value=Counter(y).most_common(1)[0][0])
            else:
                node.children[value] = self._build_tree(subset_X, subset_y, depth + 1)
        
        return node
    
    def predict(self, X):
        """预测新样本"""
        return X.apply(self._predict_sample, axis=1)
    
    def _predict_sample(self, sample):
        """预测单个样本"""
        node = self.root
        
        while not node.is_leaf:
            feature_value = sample[node.feature]
            if feature_value in node.children:
                node = node.children[feature_value]
            else:
                # 如果遇到未知的特征值，返回决策路径上最常见的类别
                # 这里简化处理，返回根节点下最常见的类别
                return self._get_most_common_class(self.root)
        
        return node.value
    
    def _get_most_common_class(self, node):
        """获取节点下最常见的类别"""
        # 这个方法用于处理未知特征值的情况
        # 简化处理，返回最常见的类别
        # 在实际应用中可能需要更复杂的处理
        return list(node.children.values())[0].value
    
    def print_tree(self, node=None, depth=0):
        """打印决策树结构"""
        if node is None:
            node = self.root
        
        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}Leaf: {node.value}")
        else:
            print(f"{indent}Feature: {node.feature}")
            for value, child in node.children.items():
                print(f"{indent}  Value: {value}")
                self.print_tree(child, depth + 2)
    
    def prune(self, X_val, y_val):
        """后剪枝"""
        self._prune_tree(self.root, X_val, y_val)
    
    def _prune_tree(self, node, X_val, y_val):
        """递归剪枝"""
        # 如果是叶节点，不需要剪枝
        if node.is_leaf:
            return node
        
        # 剪枝所有子节点
        for value, child in node.children.items():
            node.children[value] = self._prune_tree(child, X_val, y_val)
        
        # 检查是否可以剪枝当前节点
        # 如果当前节点的所有子节点都是叶节点
        if all(child.is_leaf for child in node.children.values()):
            # 计算剪枝前的验证集准确率
            original_accuracy = self._calculate_accuracy(X_val, y_val)
            
            # 保存当前节点的信息
            original_feature = node.feature
            original_children = node.children
            original_is_leaf = node.is_leaf
            original_value = node.value
            
            # 尝试剪枝（将节点变为叶节点）
            # 计算验证集中通过该节点的样本的多数类别
            node.is_leaf = True
            node.feature = None
            node.children = {}
            # 使用验证集中符合条件的样本的多数类别作为叶节点值
            mask = X_val[original_feature].isin(original_children.keys())
            if mask.any():
                node.value = Counter(y_val[mask]).most_common(1)[0][0]
            else:
                # 如果没有符合条件的样本，使用随机子节点的值
                node.value = list(original_children.values())[0].value
            
            # 计算剪枝后的验证集准确率
            pruned_accuracy = self._calculate_accuracy(X_val, y_val)
            
            # 如果剪枝后准确率下降，恢复原来的节点
            if pruned_accuracy < original_accuracy:
                node.feature = original_feature
                node.children = original_children
                node.is_leaf = original_is_leaf
                node.value = original_value
        
        return node
    
    def _calculate_accuracy(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return (predictions == y).mean()

def load_data():
    """加载训练和测试数据"""
    # 加载训练数据
    train_data = pd.read_csv('train.csv')
    # 加载预测数据
    predict_data = pd.read_csv('predict.csv')
    return train_data, predict_data

def main():
    # 加载数据
    train_data, predict_data = load_data()
    
    print("训练数据预览:")
    print(train_data.head())
    print("\n预测数据预览:")
    print(predict_data.head())
    
    # 分离特征和目标变量
    X_train = train_data.drop(columns=['weather'])
    y_train = train_data['weather']
    
    # 划分训练集和验证集用于剪枝
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )
    
    # 构建并训练ID3决策树
    print("\n构建ID3决策树...")
    tree = ID3DecisionTree(max_depth=5)
    tree.fit(X_train_sub, y_train_sub)
    
    # 打印决策树结构
    print("\n未剪枝决策树结构:")
    tree.print_tree()
    
    # 在验证集上评估未剪枝的树
    train_accuracy = tree._calculate_accuracy(X_train_sub, y_train_sub)
    val_accuracy = tree._calculate_accuracy(X_val, y_val)
    print(f"\n未剪枝决策树性能:")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"验证集准确率: {val_accuracy:.4f}")
    
    # 剪枝
    print("\n进行后剪枝...")
    tree.prune(X_val, y_val)
    
    # 打印剪枝后的决策树结构
    print("\n剪枝后决策树结构:")
    tree.print_tree()
    
    # 在验证集上评估剪枝后的树
    pruned_train_accuracy = tree._calculate_accuracy(X_train_sub, y_train_sub)
    pruned_val_accuracy = tree._calculate_accuracy(X_val, y_val)
    print(f"\n剪枝后决策树性能:")
    print(f"训练集准确率: {pruned_train_accuracy:.4f}")
    print(f"验证集准确率: {pruned_val_accuracy:.4f}")
    
    # 使用剪枝后的树进行预测
    print("\n对预测数据进行预测...")
    predictions = tree.predict(predict_data)
    
    # 添加预测结果到原始数据
    result_data = predict_data.copy()
    result_data['predicted_weather'] = predictions
    
    print("\n预测结果:")
    print(result_data)
    
    # 保存结果到CSV文件
    result_data.to_csv('result.csv', index=False)
    print("\n预测结果已保存到 result.csv")

if __name__ == "__main__":
    main()