# Category子图和双路召回功能

## 概述

本实现为mem0添加了category子图功能，实现了双路召回机制，包括：
1. **Entity子图**：传统的基于embedding的实体关系图
2. **Category子图**：基于用户画像的层次化类别图

## 架构设计

### Category子图结构
```
user -> topic -> sub_topic -> memo
```

例如：
```
Tom -> address -> home_is -> new jersey
Tom -> address -> school_is -> livington primary school, new jersey  
Tom -> birth -> age_is -> 9
```

### 双路召回机制

1. **Category召回（Context-aware）**：
   - 使用LLM分析查询，识别相关类别
   - 在category子图中搜索匹配的层次结构
   - 适用于需要上下文理解的查询

2. **Entity召回（Embedding-based）**：
   - 传统的向量相似度搜索
   - 在entity子图中查找相关实体关系
   - 适用于精确匹配的查询

## 文件结构

### 新增文件
- `mem0/graphs/category_tools.py` - Category提取和分类工具
- `mem0/graphs/extract_category.py` - 简化的category提取逻辑
- `test_category_graph.py` - 完整功能测试脚本
- `simple_test.py` - 基础功能测试脚本

### 修改文件
- `mem0/memory/memgraph_memory.py` - 增强的Memgraph实现
- `mem0/memory/main.py` - 更新search方法支持双路召回

## 核心功能

### 1. Category提取
```python
# 从用户数据中提取category信息
categories = memory._extract_categories_from_data(data, filters)
```

### 2. Category子图构建
```python
# 构建层次化的category子图
result = memory._add_category_hierarchy(
    user_id, agent_id, topic, sub_topic, memo,
    topic_embedding, sub_topic_embedding, memo_embedding
)
```

### 3. 双路召回搜索
```python
# 搜索返回两种结果
search_result = memory.search(query, user_id="user", agent_id="agent")
# 结果包含：
# - entity_relations: 实体子图结果
# - category_relations: 类别子图结果
```

## 使用示例

### 基本使用
```python
from mem0 import Memory
from mem0.config import MemoryConfig

# 配置Memgraph
config = MemoryConfig(
    graph_store={
        "provider": "memgraph",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "memgraph", 
            "password": "memgraph"
        }
    }
)

memory = Memory(config=config)

# 添加记忆（自动构建两个子图）
result = memory.add(
    "Tom lives in New Jersey and he studies in Livingston Primary School. He is 9 years old.",
    user_id="test_user",
    agent_id="test_agent"
)

# 双路召回搜索
search_result = memory.search(
    "Tell me some swimming clubs near me",
    user_id="test_user",
    agent_id="test_agent"
)

print("实体子图结果:", search_result.get('entity_relations', []))
print("类别子图结果:", search_result.get('category_relations', []))
```

### 查询分类示例

**查询**: "Tell me some swimming clubs near"
- **Category分析**: 识别出 `address` 和 `age` 类别
- **Category召回**: 在category子图中查找地址和年龄相关信息
- **Entity召回**: 在entity子图中查找游泳俱乐部相关实体

## 技术细节

### LLM工具
- `EXTRACT_CATEGORIES_TOOL`: 提取用户画像类别
- `CLASSIFY_QUERY_CATEGORIES_TOOL`: 查询类别分类

### 数据库索引
- `memzero`: 实体子图向量索引
- `memzero_category`: 类别子图向量索引

### 关系类型
- `HAS_TOPIC`: 用户 -> 主题
- `HAS_SUB_TOPIC`: 主题 -> 子主题  
- `HAS_MEMO`: 子主题 -> 记忆

## 测试

运行基础测试：
```bash
python simple_test.py
```

运行完整功能测试：
```bash
python test_category_graph.py
```

## 优势

1. **上下文感知**: Category子图提供更好的上下文理解
2. **层次化组织**: 信息按主题层次组织，便于管理
3. **双路召回**: 结合精确匹配和语义理解
4. **向后兼容**: 保持与现有API的兼容性
5. **可扩展性**: 易于添加新的类别和关系类型

## 注意事项

1. 需要Memgraph数据库支持
2. 需要配置LLM用于category提取和查询分类
3. 建议在生产环境中调整相似度阈值
4. 大量数据时考虑批量处理优化
