
"""
本地版本的Memory SEARCH - 使用本地Memory而不是MemoryClient
"""
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# 确保导入本地版本的mem0
sys.path.insert(0, '/home/azureuser/mem0')

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import Memory

load_dotenv()


def get_local_memory_config():
    """获取本地Memory配置 - 从test_category_graph.py复制的能工作的配置"""
    # 确保设置Google API Key环境变量
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "AIzaSyDLZHwabgwDySAeE9GaUb1kJEM4VgkiImc"
    
    return {
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
                "model": "gemini-2.5-flash-lite",
                "max_tokens": 8000,  # 设置合理的输出token限制，避免MAX_TOKENS错误
                "temperature": 0.1,
            }
        },
        "vector_store": {
            "provider": "elasticsearch",
            "config": {
                "collection_name": "mem_test_mocked",  # 使用test_category_graph.py中的collection名
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


class MemorySearch:
    def __init__(self, output_path="results.json", top_k=10, filter_memories=False, 
                 is_graph=False, use_dual_recall=False):
        # 使用本地Memory
        config = get_local_memory_config()
        self.memory = Memory.from_config(config)
        
        self.top_k = top_k
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.use_dual_recall = use_dual_recall

    def search_memory(self, user_id, agent_id, query, max_retries=3, retry_delay=1):
        """搜索记忆 - 使用本地Memory"""
        start_time = time.time()
        retries = 0
        
        while retries < max_retries:
            try:
                # 使用本地Memory的search方法
                search_result = self.memory.search(
                    query,
                    user_id=user_id,
                    agent_id=agent_id,
                    limit=self.top_k,
                    top_k_subtopics=5 if self.use_dual_recall else None
                )
                break
            except Exception as e:
                print(f"  搜索重试 ({retries+1}/{max_retries}): {e}")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        search_time = end_time - start_time
        
        # 解析结果
        memories = search_result.get('results', [])
        entity_relations = search_result.get('entity_relations', [])
        category_relations = search_result.get('category_relations', [])
        category_memories = search_result.get('category_related_memories', [])
        
        return {
            'memories': memories,
            'entity_relations': entity_relations,
            'category_relations': category_relations,
            'category_memories': category_memories,
            'search_time': search_time
        }

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, agent_id, question, answer, category):
        """搜索并记录结果"""
        # 搜索speaker 1
        result_1 = self.search_memory(speaker_1_user_id, agent_id, question)
        
        # 搜索speaker 2  
        result_2 = self.search_memory(speaker_2_user_id, agent_id, question)
        
        # 格式化结果
        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "speaker_1_memories": result_1['memories'][:5],
            "speaker_2_memories": result_2['memories'][:5],
            "num_speaker_1_memories": len(result_1['memories']),
            "num_speaker_2_memories": len(result_2['memories']),
            "speaker_1_memory_time": result_1['search_time'],
            "speaker_2_memory_time": result_2['search_time'],
        }
        
        if self.use_dual_recall:
            # 双路召回的额外字段
            result.update({
                "speaker_1_entity_relations": result_1['entity_relations'][:10],
                "speaker_2_entity_relations": result_2['entity_relations'][:10],
                "speaker_1_category_relations": result_1['category_relations'][:10],
                "speaker_2_category_relations": result_2['category_relations'][:10],
                "speaker_1_category_memories": result_1['category_memories'][:10],
                "speaker_2_category_memories": result_2['category_memories'][:10],
            })
        else:
            # 原始graph模式
            result.update({
                "speaker_1_graph_memories": result_1['entity_relations'][:10],
                "speaker_2_graph_memories": result_2['entity_relations'][:10],
            })
        
        return result

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id, agent_id):
        """处理单个问题"""
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)

        print(f"    问题: {question[:60]}...")
        
        result = self.answer_question(
            speaker_a_user_id, speaker_b_user_id, agent_id,
            question, answer, category
        )
        
        # 保存结果
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        
        return result

    def process_data_file(self, file_path):
        """处理数据文件"""
        with open(file_path, "r") as f:
            data = json.load(f)
        
        agent_id = "locomo_eval"

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            print(f"\n{'='*80}")
            print(f"对话 {idx+1}/{len(data)}: {speaker_a} <-> {speaker_b}")
            print(f"{'='*80}")

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Questions for conversation {idx}", leave=False
            ):
                # 跳过category 5
                if str(question_item.get('category', '')) == '5':
                    continue
                
                try:
                    result = self.process_question(
                        question_item, speaker_a_user_id, speaker_b_user_id, agent_id
                    )
                    self.results[idx].append(result)
                except Exception as e:
                    print(f"    ✗ 问题处理失败: {e}")
                    continue

            # 保存中间结果
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)

        # 最终保存
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\n✓ 所有结果已保存到: {self.output_path}")
