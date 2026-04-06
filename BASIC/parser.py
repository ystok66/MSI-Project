"""
Parser: 组合/作用域层 (Layer 3)

将词序列解析为候选 AST (Abstract Syntax Tree):
- PRIM(word): 名词, 产生颜色
- APPLY(op, arg): 算子应用到参数上
- CONCAT(children): 序列拼接

枚举策略:
1. 根据词类型 (PRIM/OP/UNKNOWN) 生成候选类型分配
2. 对每种类型分配, 用右优先结合律生成 parse 树
3. 返回所有候选 parse, 由 RSA 评分选最优
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
import numpy as np
from itertools import product


# ── AST 节点 ──

@dataclass
class ParseNode:
    """AST 节点."""
    kind: str               # 'prim', 'apply', 'concat'
    token: Optional[str]     # 关联的 token (prim/apply)
    children: List['ParseNode'] = field(default_factory=list)
    
    def __repr__(self):
        if self.kind == 'prim':
            return self.token
        elif self.kind == 'apply':
            args_str = ', '.join(str(c) for c in self.children)
            return f"{self.token}({args_str})"
        else:  # concat
            return ' '.join(str(c) for c in self.children)

    def depth(self):
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)
    
    def size(self):
        return 1 + sum(c.size() for c in self.children)


# ── 词类型 ──

PRIM = 'prim'    # 名词: 产生颜色
OP = 'op'        # 算子: 变换序列
UNK = 'unknown'  # 未知


def infer_word_types(words: List[str], 
                     known_prims: Set[str],
                     known_ops: Set[str]) -> Dict[str, str]:
    """
    推断每个词的类型.
    
    规则:
    - 已知名词 → PRIM
    - 已知算子 → OP
    - 未知 → UNK (会在枚举中尝试两种)
    """
    types = {}
    for w in words:
        w_lower = w.lower()
        if w_lower in known_prims:
            types[w] = PRIM
        elif w_lower in known_ops:
            types[w] = OP
        else:
            types[w] = UNK
    return types


# ── Parse 枚举 ──

def enumerate_parses(words: List[str],
                     word_types: Dict[str, str],
                     max_parses: int = 50) -> List[ParseNode]:
    """
    枚举候选 parse 树.
    
    策略:
    1. 对 UNK 词, 尝试 PRIM 和 OP 两种类型
    2. 对每种类型分配, 生成 parse:
       - 全是 PRIM → CONCAT
       - 有 OP → APPLY 把 OP 和相邻 PRIM 结合
    3. 支持多种结合方式:
       - 右结合: op arg → APPLY(op, arg)  
       - 左结合: arg op → APPLY(op, arg)
       - 后缀: args op → APPLY(op, CONCAT(args))
    """
    parses = []
    
    # Step 1: 生成所有可能的类型分配
    unk_positions = [i for i, w in enumerate(words) if word_types.get(w, UNK) == UNK]
    
    if len(unk_positions) > 4:
        # 太多未知 → 只尝试 "全PRIM" 和 "单OP" 模式
        type_assignments = [_all_prim_assignment(words, word_types)]
        for pos in unk_positions:
            asgn = _all_prim_assignment(words, word_types)
            asgn[pos] = OP
            type_assignments.append(asgn)
    else:
        # 枚举所有 UNK 的 PRIM/OP 组合
        type_assignments = []
        for combo in product([PRIM, OP], repeat=len(unk_positions)):
            asgn = []
            unk_idx = 0
            for i, w in enumerate(words):
                t = word_types.get(w, UNK)
                if t == UNK:
                    asgn.append(combo[unk_idx])
                    unk_idx += 1
                else:
                    asgn.append(t)
            type_assignments.append(asgn)
    
    # Step 2: 对每种类型分配, 生成 parse 树
    seen = set()
    for types in type_assignments:
        for parse in _generate_parses_for_types(words, types):
            key = str(parse)
            if key not in seen:
                seen.add(key)
                parses.append(parse)
                if len(parses) >= max_parses:
                    return parses
    
    return parses


def _all_prim_assignment(words, word_types):
    """全部当名词的类型分配."""
    asgn = []
    for w in words:
        t = word_types.get(w, UNK)
        if t == UNK:
            asgn.append(PRIM)
        else:
            asgn.append(t)
    return asgn


def _generate_parses_for_types(words: List[str], 
                                types: List[str]) -> List[ParseNode]:
    """
    给定固定的词类型分配, 生成所有合法 parse.
    
    核心逻辑:
    1. 全 PRIM → CONCAT
    2. 有 OP → 尝试多种结合方式:
       a. 前缀: OP arg → APPLY(OP, arg)
       b. 后缀: arg OP → APPLY(OP, arg)  
       c. 中缀: arg1 OP arg2 → APPLY(OP, [arg1, arg2])
       d. 全局: OP(所有前面的PRIM) / OP(所有后面的PRIM)
    """
    parses = []
    n = len(words)
    
    op_positions = [i for i in range(n) if types[i] == OP]
    prim_positions = [i for i in range(n) if types[i] == PRIM]
    
    # Case 1: 没有 OP → 全部 CONCAT
    if not op_positions:
        children = [ParseNode(kind='prim', token=words[i]) for i in range(n)]
        if len(children) == 1:
            parses.append(children[0])
        else:
            parses.append(ParseNode(kind='concat', token=None, children=children))
        return parses
    
    # Case 2: 有 OP → 多种解析方式
    
    # Strategy A: 右结合 — 每个 OP 作用于紧邻右边的所有 PRIM
    parse_a = _right_associative_parse(words, types)
    if parse_a:
        parses.append(parse_a)
    
    # Strategy B: 左结合 — 每个 OP 作用于紧邻左边的所有 PRIM
    parse_b = _left_associative_parse(words, types)
    if parse_b:
        parses.append(parse_b)
    
    # Strategy C: 中缀 — OP 在两个 PRIM 之间, 作用于两者
    parse_c = _infix_parse(words, types)
    if parse_c:
        parses.append(parse_c)
    
    # Strategy D: 全局后缀 — OP 在最后, 作用于前面所有 PRIM
    parse_d = _suffix_global_parse(words, types)
    if parse_d:
        parses.append(parse_d)
    
    # Strategy E: PRIM 透传 (OP被忽略, 当作 identity)
    prims_only = [ParseNode(kind='prim', token=words[i]) for i in prim_positions]
    if prims_only:
        if len(prims_only) == 1:
            parses.append(prims_only[0])
        else:
            parses.append(ParseNode(kind='concat', token=None, children=prims_only))
    
    return parses


def _right_associative_parse(words, types):
    """
    右结合: 从左到右, 遇到 OP 时把右边的 PRIM 作为参数.
    例: a OP b c → CONCAT(a, APPLY(OP, CONCAT(b, c)))
    """
    n = len(words)
    nodes = []
    i = 0
    while i < n:
        if types[i] == OP:
            # 收集右边所有连续 PRIM 作为参数
            args = []
            j = i + 1
            while j < n and types[j] == PRIM:
                args.append(ParseNode(kind='prim', token=words[j]))
                j += 1
            
            # 也包括右边下一个 OP 的递归结果
            if not args:
                # OP 后面没有 PRIM → 跳过 (可能是前缀)
                nodes.append(ParseNode(kind='prim', token=words[i]))  # 当 PRIM
                i += 1
            else:
                arg = args[0] if len(args) == 1 else \
                      ParseNode(kind='concat', token=None, children=args)
                nodes.append(ParseNode(kind='apply', token=words[i], 
                                       children=[arg]))
                i = j
        else:
            nodes.append(ParseNode(kind='prim', token=words[i]))
            i += 1
    
    if len(nodes) == 1:
        return nodes[0]
    return ParseNode(kind='concat', token=None, children=nodes)


def _left_associative_parse(words, types):
    """
    左结合: 遇到 OP 时, 把左边已积累的 PRIM 作为参数.
    例: a b OP c → CONCAT(APPLY(OP, CONCAT(a, b)), c)
    """
    n = len(words)
    nodes = []
    pending_prims = []
    
    for i in range(n):
        if types[i] == PRIM:
            pending_prims.append(ParseNode(kind='prim', token=words[i]))
        elif types[i] == OP:
            if pending_prims:
                arg = pending_prims[0] if len(pending_prims) == 1 else \
                      ParseNode(kind='concat', token=None, children=pending_prims)
                nodes.append(ParseNode(kind='apply', token=words[i],
                                       children=[arg]))
                pending_prims = []
            else:
                nodes.append(ParseNode(kind='prim', token=words[i]))
    
    nodes.extend(pending_prims)
    
    if len(nodes) == 1:
        return nodes[0]
    if not nodes:
        return None
    return ParseNode(kind='concat', token=None, children=nodes)


def _infix_parse(words, types):
    """
    中缀: OP 取左右各一个 PRIM 作为参数.
    例: a OP b → APPLY(OP, [a, b])
    """
    n = len(words)
    nodes = []
    i = 0
    used = set()
    
    # 先找 infix 模式
    for i in range(n):
        if types[i] == OP and i > 0 and i < n-1:
            if types[i-1] == PRIM and types[i+1] == PRIM:
                if i-1 not in used and i+1 not in used:
                    left = ParseNode(kind='prim', token=words[i-1])
                    right = ParseNode(kind='prim', token=words[i+1])
                    nodes.append((i, ParseNode(kind='apply', token=words[i],
                                               children=[left, right])))
                    used.update({i-1, i, i+1})
    
    if not used:
        return None
    
    # 加入未使用的词
    result = []
    for i in range(n):
        if i in used:
            # 检查是否是 OP 的位置 (已处理)
            for pos, node in nodes:
                if pos == i:
                    result.append(node)
        else:
            result.append(ParseNode(kind='prim', token=words[i]))
    
    if len(result) == 1:
        return result[0]
    return ParseNode(kind='concat', token=None, children=result)


def _suffix_global_parse(words, types):
    """
    全局后缀: 最后一个 OP 作用于前面所有内容.
    例: a b c OP → APPLY(OP, CONCAT(a, b, c))
    """
    n = len(words)
    if n < 2:
        return None
    
    # 找最后一个 OP
    last_op = None
    for i in range(n-1, -1, -1):
        if types[i] == OP:
            last_op = i
            break
    
    if last_op is None or last_op == 0:
        return None
    
    # 收集 OP 之前的所有词
    before = []
    for i in range(last_op):
        before.append(ParseNode(kind='prim', token=words[i]))
    
    if not before:
        return None
    
    arg = before[0] if len(before) == 1 else \
          ParseNode(kind='concat', token=None, children=before)
    
    result = ParseNode(kind='apply', token=words[last_op], children=[arg])
    
    # OP 之后还有词就 concat
    after = []
    for i in range(last_op + 1, n):
        after.append(ParseNode(kind='prim', token=words[i]))
    
    if after:
        return ParseNode(kind='concat', token=None, children=[result] + after)
    
    return result
