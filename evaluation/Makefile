
# Run the experiments

# ============================================================================
# 本地版本（使用本地Memory，不需要Mem0 API key）
# ============================================================================
run-mem0-local-add:
	python run_experiments.py --technique_type mem0 --method add --use_local --is_graph

run-mem0-local-search:
	python run_experiments.py --technique_type mem0 --method search --use_local --output_folder results/ --top_k 30

run-mem0-local-plus-search:
	python run_experiments.py --technique_type mem0 --method search --use_local --is_graph --output_folder results/ --top_k 30

run-mem0-local-dual-recall-search:
	python run_experiments.py --technique_type mem0 --method search --use_local --use_dual_recall --output_folder results/ --top_k 30 --top_k_subtopics 5

run-mem0-local-dual-recall-search-k3:
	python run_experiments.py --technique_type mem0 --method search --use_local --use_dual_recall --output_folder results/ --top_k 30 --top_k_subtopics 3

run-mem0-local-dual-recall-search-k10:
	python run_experiments.py --technique_type mem0 --method search --use_local --use_dual_recall --output_folder results/ --top_k 30 --top_k_subtopics 10

# ============================================================================
# 云端版本（需要Mem0 API key）
# ============================================================================
run-mem0-add:
	python run_experiments.py --technique_type mem0 --method add

run-mem0-search:
	python run_experiments.py --technique_type mem0 --method search --output_folder results/ --top_k 30

run-mem0-plus-add:
	python run_experiments.py --technique_type mem0 --method add --is_graph

run-mem0-plus-search:
	python run_experiments.py --technique_type mem0 --method search --is_graph --output_folder results/ --top_k 30

run-rag:
	python run_experiments.py --technique_type rag --chunk_size 500 --num_chunks 1 --output_folder results/

run-full-context:
	python run_experiments.py --technique_type rag --chunk_size -1 --num_chunks 1 --output_folder results/

run-langmem:
	python run_experiments.py --technique_type langmem --output_folder results/

run-zep-add:
	python run_experiments.py --technique_type zep --method add --output_folder results/

run-zep-search:
	python run_experiments.py --technique_type zep --method search --output_folder results/

run-openai:
	python run_experiments.py --technique_type openai --output_folder results/
