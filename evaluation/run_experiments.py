import argparse
import os

from src.utils import METHODS, TECHNIQUES

# 确保设置Google API Key环境变量
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDLZHwabgwDySAeE9GaUb1kJEM4VgkiImc"

# 使用延迟导入，避免未安装的模块导致整个脚本失败


class Experiment:
    def __init__(self, technique_type, chunk_size):
        self.technique_type = technique_type
        self.chunk_size = chunk_size

    def run(self):
        print(f"Running experiment with technique: {self.technique_type}, chunk size: {self.chunk_size}")


def main():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument("--technique_type", choices=TECHNIQUES, default="mem0", help="Memory technique to use")
    parser.add_argument("--method", choices=METHODS, default="add", help="Method to use")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")
    parser.add_argument("--output_folder", type=str, default="results/", help="Output path for results")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top memories to retrieve")
    parser.add_argument("--filter_memories", action="store_true", default=False, help="Whether to filter memories")
    parser.add_argument("--is_graph", action="store_true", default=False, help="Whether to use graph-based search")
    parser.add_argument("--use_dual_recall", action="store_true", default=False, help="Whether to use dual recall (entity + category)")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks to process")
    parser.add_argument("--top_k_subtopics", type=int, default=5, help="Number of top subtopics for category search")
    parser.add_argument("--use_local", action="store_true", default=False, help="Use local Memory instead of MemoryClient")

    args = parser.parse_args()

    # Add your experiment logic here
    print(f"Running experiments with technique: {args.technique_type}, chunk size: {args.chunk_size}")

    if args.technique_type == "mem0":
        # 根据use_local参数选择使用本地还是云端版本
        if args.use_local:
            print("使用本地Memory版本（不需要API key）")
            from src.memzero.add_local import MemoryADD
            from src.memzero.search_local import MemorySearch
        else:
            print("使用MemoryClient版本（需要API key）")
            from src.memzero.add import MemoryADD
            from src.memzero.search import MemorySearch
        
        if args.method == "add":
            memory_manager = MemoryADD(data_path="dataset/locomo10.json", is_graph=args.is_graph)
            memory_manager.process_all_conversations()
        elif args.method == "search":
            if args.use_dual_recall:
                output_file_path = os.path.join(
                    args.output_folder,
                    f"mem0_results_top_{args.top_k}_dual_recall_subtopics_{args.top_k_subtopics}_local.json",
                )
            else:
                output_file_path = os.path.join(
                    args.output_folder,
                    f"mem0_results_top_{args.top_k}_filter_{args.filter_memories}_graph_{args.is_graph}_local.json",
                )
            memory_searcher = MemorySearch(
                output_file_path, 
                args.top_k, 
                args.filter_memories, 
                args.is_graph, 
                args.use_dual_recall
            )
            memory_searcher.process_data_file("dataset/locomo10.json")
    elif args.technique_type == "rag":
        from src.rag import RAGManager
        output_file_path = os.path.join(args.output_folder, f"rag_results_{args.chunk_size}_k{args.num_chunks}.json")
        rag_manager = RAGManager(data_path="dataset/locomo10_rag.json", chunk_size=args.chunk_size, k=args.num_chunks)
        rag_manager.process_all_conversations(output_file_path)
    elif args.technique_type == "langmem":
        from src.langmem import LangMemManager
        output_file_path = os.path.join(args.output_folder, "langmem_results.json")
        langmem_manager = LangMemManager(dataset_path="dataset/locomo10_rag.json")
        langmem_manager.process_all_conversations(output_file_path)
    elif args.technique_type == "zep":
        from src.zep.add import ZepAdd
        from src.zep.search import ZepSearch
        if args.method == "add":
            zep_manager = ZepAdd(data_path="dataset/locomo10.json")
            zep_manager.process_all_conversations("1")
        elif args.method == "search":
            output_file_path = os.path.join(args.output_folder, "zep_search_results.json")
            zep_manager = ZepSearch()
            zep_manager.process_data_file("dataset/locomo10.json", "1", output_file_path)
    elif args.technique_type == "openai":
        from src.openai.predict import OpenAIPredict
        output_file_path = os.path.join(args.output_folder, "openai_results.json")
        openai_manager = OpenAIPredict()
        openai_manager.process_data_file("dataset/locomo10.json", output_file_path)
    else:
        raise ValueError(f"Invalid technique type: {args.technique_type}")


if __name__ == "__main__":
    main()