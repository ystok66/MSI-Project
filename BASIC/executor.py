"""
Executor: 程序执行层 (Layer 4)

给定 parse tree (AST), 递归执行, 产生预测输出:
    - PRIM(word): 查 Lexicon, 返回颜色
    - APPLY(op, arg): 先执行 arg 得到序列, 再用 Operator 变换
    - CONCAT(children): 依次执行子节点, 拼接结果

同时计算 likelihood: P(output | input, program)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from parser import ParseNode
from operator_table import OperatorTable
from concepts import ConceptTable


def execute_parse(node: ParseNode,
                  lexicon: Dict[str, str],
                  operators: OperatorTable,
                  color_vecs: Dict[str, np.ndarray]) -> List[str]:
    """
    递归执行 parse 树, 返回颜色序列.
    
    Args:
        node: AST 节点
        lexicon: word → color_name 词典
        operators: 算子表
        color_vecs: color_name → feature vector
    
    Returns:
        颜色名列表, e.g. ['BLUE', 'GREEN', 'BLUE']
    """
    if node.kind == 'prim':
        # 原子: 查词典
        color = lexicon.get(node.token, None)
        if color:
            return [color]
        else:
            return ['?']
    
    elif node.kind == 'concat':
        # 拼接: 依次执行
        result = []
        for child in node.children:
            result.extend(execute_parse(child, lexicon, operators, color_vecs))
        return result
    
    elif node.kind == 'apply':
        # 算子应用
        op_token = node.token
        
        if not node.children:
            return ['?']
        
        # 先执行参数
        arg_colors = []
        for child in node.children:
            arg_colors.extend(
                execute_parse(child, lexicon, operators, color_vecs))
        
        # 用算子变换
        if not operators.has(op_token):
            # 未知算子 → 透传参数
            return arg_colors
        
        op = operators.get(op_token)
        
        # 把颜色序列转成向量, 执行变换, 再解码
        arg_vecs = [color_vecs.get(c, np.zeros(len(next(iter(color_vecs.values())))))
                    for c in arg_colors]
        
        result_vecs = _apply_operator_to_sequence(op, arg_vecs, color_vecs)
        
        # 解码: 向量 → 最近颜色
        result_colors = [_vec_to_color(v, color_vecs) for v in result_vecs]
        return result_colors
    
    return ['?']


def _apply_operator_to_sequence(op, arg_vecs: List[np.ndarray],
                                 color_vecs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    用算子的 A 矩阵和 b 向量变换输入序列.
    
    支持变长输出:
    - 如果 A 的结构暗示重复/扩展, 输出可以更长
    
    简化实现: 
    - 用 A 的列块分析输出长度
    - 或直接逐元素变换
    """
    d = len(arg_vecs[0]) if arg_vecs else 6
    n_in = len(arg_vecs)
    
    if n_in == 0:
        return []
    
    # 将输入序列展平为一个大向量
    max_slots = op.A_mu.shape[1] // d if d > 0 else 1
    x_flat = np.zeros(op.A_mu.shape[1])
    for i, v in enumerate(arg_vecs):
        if i < max_slots:
            x_flat[i*d:(i+1)*d] = v
    
    # 执行 y = A @ x + b
    y_flat = op.A_mu @ x_flat + op.b_mu
    
    # 分割输出向量为颜色块
    n_out_slots = op.A_mu.shape[0] // d if d > 0 else 1
    result = []
    for i in range(n_out_slots):
        v = y_flat[i*d:(i+1)*d]
        # 如果向量几乎为零, 跳过 (空 slot)
        if np.linalg.norm(v) > 0.1:
            result.append(v)
    
    if not result:
        # 退化: 至少返回输入
        return arg_vecs
    
    return result


def _vec_to_color(vec: np.ndarray, color_vecs: Dict[str, np.ndarray]) -> str:
    """向量 → 最近颜色名."""
    best_color = '?'
    best_dist = np.inf
    for name, ref in color_vecs.items():
        dist = np.linalg.norm(vec - ref)
        if dist < best_dist:
            best_dist = dist
            best_color = name
    return best_color


def program_likelihood(node: ParseNode,
                       lexicon: Dict[str, str],
                       operators: OperatorTable,
                       color_vecs: Dict[str, np.ndarray],
                       expected_output: List[str],
                       sigma: float = 0.5) -> float:
    """
    计算 P(output | program).
    
    P = exp(-||y_pred - y_true||^2 / (2σ²))
    
    同时考虑长度匹配:
    - 长度不匹配有额外惩罚
    - 颜色匹配用精确匹配计数
    
    Returns:
        log likelihood (负数, 越大越好)
    """
    predicted = execute_parse(node, lexicon, operators, color_vecs)
    
    # 长度惩罚
    len_penalty = -abs(len(predicted) - len(expected_output)) * 2.0
    
    # 颜色匹配
    match_score = 0.0
    min_len = min(len(predicted), len(expected_output))
    for i in range(min_len):
        if predicted[i] == expected_output[i]:
            match_score += 1.0
        else:
            match_score -= 1.0
    
    # 归一化
    max_len = max(len(predicted), len(expected_output), 1)
    normalized = match_score / max_len
    
    # Log likelihood
    log_lik = (normalized + len_penalty / max_len) / sigma
    
    return log_lik


def score_program_on_examples(node: ParseNode,
                              lexicon: Dict[str, str],
                              operators: OperatorTable,
                              color_vecs: Dict[str, np.ndarray],
                              examples: List[Dict],
                              sigma: float = 0.5) -> float:
    """
    在多个例子上评估 program 的总 log likelihood.
    """
    total = 0.0
    for ex in examples:
        total += program_likelihood(
            node, lexicon, operators, color_vecs,
            ex['output'], sigma=sigma
        )
    return total
