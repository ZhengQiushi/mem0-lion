#!/usr/bin/env python3
"""
测试增强的类别召回功能：使用embedding匹配sub_topic
"""

import os
import sys
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, '/home/azureuser/mem0')

from mem0 import Memory

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["GOOGLE_API_KEY"] = "AIzaSyDLZHwabgwDySAeE9GaUb1kJEM4VgkiImc"

def test_enhanced_category_search():
    """测试增强的类别召回功能"""
    
    # 配置Memgraph
    config = {
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": "gemini-embedding-001",
                "embedding_dims": 1536,
            }
        },
        "llm": {
            "provider": "gemini",
            "config": {
                "model": "gemini-2.5-flash",
            }
        },
        "vector_store": {
            "provider": "elasticsearch",
            "config": {
                "collection_name": "mem_test_enhanced_search",
                "host": "https://ec6d45523ca143c38f43de6dd9257d94.westus2.azure.elastic-cloud.com",
                "port": 443,
                "embedding_model_dims": 1536,
                "api_key": "b0lCNW5aY0IyOUdoQ0VUVVVFSlY6OExqekM3MTZTV21PS2VUb3VNaV9XZw==",
            }
        },
        "graph_store": {
            "provider": "memgraph",
            "config": {
                "url": "bolt://localhost:7687",
                "username": "memgraph",
                "password": "xxx"
            }
        }
    }
    
    # 创建Memory实例
    memory = Memory.from_config(config)
    
    # 测试数据
    test_data = """
    Tom lives in New Jersey and he studies in Livingston Primary School. 
    He is 9 years old and loves playing soccer and swimming. 
    His favorite color is blue and he likes pizza.
    He also enjoys reading science fiction books.
    """
    
    print("=== 测试增强的类别召回功能 ===")
    print(f"输入数据: {test_data}")
    
    try:
        # 添加记忆
        result = memory.add(
            test_data,
            user_id="test_user_tom_enhanced",
            agent_id="test_agent"
        )
        
        print(f"\n添加结果:")
        print(f"- 删除的实体: {len(result.get('deleted_entities', []))}")
        print(f"- 添加的实体: {len(result.get('added_entities', []))}")
        print(f"- 添加的类别: {len(result.get('added_categories', []))}")
        
        # 测试搜索功能
        print("\n=== 测试增强的双路召回搜索 ===")
        
        # 测试查询1: 关于地址和兴趣的查询
        query1 = "Tell me about swimming clubs near me"
        print(f"\n查询1: {query1}")
        print(f"使用 top_k_subtopics=3")
        
        search_result1 = memory.search(
            query1,
            user_id="test_user_tom_enhanced",
            agent_id="test_agent",
            limit=10,
            top_k_subtopics=10  # 只返回top 3个sub_topic
        )
        
        print("\n搜索结果:")
        if search_result1.get('entity_relations'):
            print(f"实体子图结果: {len(search_result1['entity_relations'])} 个关系")
            for i, relation in enumerate(search_result1['entity_relations'][:5]):
                print(f"  {i+1}. {relation}")
        else:
            print("  无实体子图结果")
            
        if search_result1.get('category_relations'):
            print(f"类别子图结果: {len(search_result1['category_relations'])} 个关系")
            for i, relation in enumerate(search_result1['category_relations']):
                print(f"  {i+1}. {relation}")
        else:
            print("  无类别子图结果")

        if search_result1.get('category_related_memories'):
            print(f"类别子图记忆结果: {len(search_result1['category_related_memories'])} 个记忆")
            for i, relation in enumerate(search_result1['category_related_memories']):
                print(f"  {i+1}. {relation}")
        else:
            print("  无类别子图记忆结果")
            
        # 测试查询2: 关于兴趣的查询
        query2 = "What sports does Tom like?"
        print(f"\n查询2: {query2}")
        print(f"使用 top_k_subtopics=5 (默认)")
        
        search_result2 = memory.search(
            query2,
            user_id="test_user_tom_enhanced", 
            agent_id="test_agent",
            limit=10,
            top_k_subtopics=10
        )
        
        print("\n搜索结果:")
        if search_result2.get('entity_relations'):
            print(f"实体子图结果: {len(search_result2['entity_relations'])} 个关系")
            for i, relation in enumerate(search_result2['entity_relations'][:5]):
                print(f"  {i+1}. {relation}")
        else:
            print("  无实体子图结果")
            
        if search_result2.get('category_relations'):
            print(f"类别子图结果: {len(search_result2['category_relations'])} 个关系")
            for i, relation in enumerate(search_result2['category_relations']):
                print(f"  {i+1}. {relation}")
        else:
            print("  无类别子图结果")

        if search_result2.get('category_related_memories'):
            print(f"类别子图记忆结果: {len(search_result2['category_related_memories'])} 个记忆")
            for i, relation in enumerate(search_result2['category_related_memories']):
                print(f"  {i+1}. {relation}")
        else:
            print("  无类别子图记忆结果")

        # 测试查询3: 关于年龄的查询
        query3 = "How old is Tom?"
        print(f"\n查询3: {query3}")
        print(f"使用 top_k_subtopics=2")
        
        search_result3 = memory.search(
            query3,
            user_id="test_user_tom_enhanced",
            agent_id="test_agent", 
            limit=10,
            top_k_subtopics=10  # 只返回top 2个sub_topic
        )
        
        print("\n搜索结果:")
        if search_result3.get('entity_relations'):
            print(f"实体子图结果: {len(search_result3['entity_relations'])} 个关系")
            for i, relation in enumerate(search_result3['entity_relations'][:5]):
                print(f"  {i+1}. {relation}")
        else:
            print("  无实体子图结果")
            
        if search_result3.get('category_relations'):
            print(f"类别子图结果: {len(search_result3['category_relations'])} 个关系")
            for i, relation in enumerate(search_result3['category_relations']):
                print(f"  {i+1}. {relation}")
        else:
            print("  无类别子图结果")


        if search_result3.get('category_related_memories'):
            print(f"类别子图记忆结果: {len(search_result3['category_related_memories'])} 个记忆")
            for i, relation in enumerate(search_result3['category_related_memories']):
                print(f"  {i+1}. {relation}")
        else:
            print("  无类别子图记忆结果")

        print("\n=== 测试完成 ===")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_category_search()




