"""
本地版本的Memory ADD - 使用本地Memory而不是MemoryClient
"""
import json
import os
import sys
import time

# 确保导入本地版本的mem0
sys.path.insert(0, '/home/azureuser/mem0')

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import Memory

load_dotenv()

# 本地Memory配置 - 从test_category_graph.py复制的能工作的配置
def get_local_memory_config():
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
                "max_tokens": 10240,
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


class MemoryADD:
    def __init__(self, data_path=None, batch_size=2, is_graph=False):
        # 使用本地Memory
        config = get_local_memory_config()
        self.memory = Memory.from_config(config)
        
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, messages, agent_id, retries=3):
        """添加记忆 - 使用本地Memory"""
        # 将messages转换为文本
        if isinstance(messages, list):
            text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages])
        else:
            text = str(messages)
        
        for attempt in range(retries):
            try:
                result = self.memory.add(
                    text,
                    user_id=user_id,
                    agent_id=agent_id
                )
                return result
            except Exception as e:
                print(f"  添加记忆失败 (尝试 {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, agent_id, desc):
        """为speaker添加记忆"""
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, agent_id)

    def process_conversation(self, item, idx):
        """处理单个对话"""
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"
        agent_id = "locomo_eval"

        print(f"\n处理对话 {idx}: {speaker_a} <-> {speaker_b}")

        # 处理所有sessions
        for key in sorted(conversation.keys()):
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            if not key.startswith("session_"):
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation.get(date_time_key, "")
            chats = conversation[key]

            messages_a = []
            messages_b = []
            
            for chat in chats:
                speaker = chat["speaker"]
                text = chat["text"]
                
                if speaker == speaker_a:
                    messages_a.append({"role": "user", "content": f"{speaker_a}: {text}"})
                    messages_b.append({"role": "assistant", "content": f"{speaker_a}: {text}"})
                elif speaker == speaker_b:
                    messages_a.append({"role": "assistant", "content": f"{speaker_b}: {text}"})
                    messages_b.append({"role": "user", "content": f"{speaker_b}: {text}"})

            # 串行添加记忆（先处理speaker_a，再处理speaker_b）
            self.add_memories_for_speaker(
                speaker_a_user_id, messages_a, agent_id, 
                f"Adding for {speaker_a}"
            )
            
            self.add_memories_for_speaker(
                speaker_b_user_id, messages_b, agent_id,
                f"Adding for {speaker_b}"
            )

        print(f"✓ 对话 {idx} 处理完成")

    def process_all_conversations(self, max_workers=1):
        """处理所有对话 - 注意：本地版本建议max_workers=1避免并发问题"""
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        
        # 本地版本使用单线程，避免并发导致的问题
        for idx, item in enumerate(self.data):
            try:
                self.process_conversation(item, idx)
            except Exception as e:
                print(f"✗ 对话 {idx} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue