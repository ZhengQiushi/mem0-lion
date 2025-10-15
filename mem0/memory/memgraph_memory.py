import logging
import json

from mem0.memory.utils import format_entities, sanitize_relationship_for_cypher

try:
    from langchain_memgraph.graphs.memgraph import Memgraph
except ImportError:
    raise ImportError("langchain_memgraph is not installed. Please install it using pip install langchain-memgraph")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    # BM25 is no longer needed in the search method as per new requirements.
    # Keeping the import block for now, but its usage will be removed.
    pass 

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.category_tools import (
    EXTRACT_CATEGORIES_TOOL,
    EXTRACT_CATEGORIES_STRUCT_TOOL,
    CLASSIFY_QUERY_CATEGORIES_TOOL,
    CLASSIFY_QUERY_CATEGORIES_STRUCT_TOOL,
    get_category_extraction_prompt,
    get_category_extraction_kwargs,
    pack_category_input,
)
from mem0.graphs.extract_category import get_default_profiles
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)

class MemoryGraph:
    def __init__(self, config):
        self.config = config
        self.graph = Memgraph(
            self.config.graph_store.config.url,
            self.config.graph_store.config.username,
            self.config.graph_store.config.password,
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            {"enable_embeddings": True},
        )

        # Default to openai if no specific provider is configured
        self.llm_provider = "openai"
        if self.config.llm and self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store and self.config.graph_store.llm and self.config.graph_store.llm.provider:
            self.llm_provider = self.config.graph_store.llm.provider

        # Get LLM config with proper null checks
        llm_config = None
        if self.config.graph_store and self.config.graph_store.llm and hasattr(self.config.graph_store.llm, "config"):
            llm_config = self.config.graph_store.llm.config
        elif hasattr(self.config.llm, "config"):
            llm_config = self.config.llm.config
        self.llm = LlmFactory.create(self.llm_provider, llm_config)
        self.user_id = None
        self.threshold = 0.7

        # Setup Memgraph:
        # 1. Create vector index (created Entity label on all nodes)
        # 2. Create label property index for performance optimizations
        embedding_dims = self.config.embedder.config["embedding_dims"]
        index_info = self._fetch_existing_indexes()
        # Create vector index if not exists for memzero
        if not any(idx.get("index_name") == "memzero" for idx in index_info["vector_index_exists"]):
            self.graph.query(
                f"CREATE VECTOR INDEX memzero ON :Entity(embedding) WITH CONFIG {{'dimension': {embedding_dims}, 'capacity': 1000, 'metric': 'cos'}};"
            )
        # --- NEW: Create vector index if not exists for memzero_category ---
        if not any(idx.get("index_name") == "memzero_category" for idx in index_info["vector_index_exists"]):
            self.graph.query(
                f"CREATE VECTOR INDEX memzero_category ON :Category(embedding) WITH CONFIG {{'dimension': {embedding_dims}, 'capacity': 1000, 'metric': 'cos'}};"
            )

        # Create label+property index if not exists for Entity
        if not any(
            idx.get("index type") == "label+property" and idx.get("label") == "Entity" and idx.get("properties") == "user_id"
            for idx in index_info["index_exists"]
        ):
            self.graph.query("CREATE INDEX ON :Entity(user_id);")
        # Create label index if not exists for Entity
        if not any(
            idx.get("index type") == "label" and idx.get("label") == "Entity" for idx in index_info["index_exists"]
        ):
            self.graph.query("CREATE INDEX ON :Entity;")
        
        # --- NEW: Create label+property index if not exists for Category ---
        if not any(
            idx.get("index type") == "label+property" and idx.get("label") == "Category" and idx.get("properties") == "user_id"
            for idx in index_info["index_exists"]
        ):
            self.graph.query("CREATE INDEX ON :Category(user_id);")
        # --- NEW: Create label index if not exists for Category ---
        if not any(
            idx.get("index type") == "label" and idx.get("label") == "Category" for idx in index_info["index_exists"]
        ):
            self.graph.query("CREATE INDEX ON :Category;")

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """
        # Extract entities for entity subgraph
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)

        # Extract categories for category subgraph
        categories = self._extract_categories_from_data(data, filters)
        
        # TODO: Batch queries with APOC plugin
        # TODO: Add more filter support
        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)
        
        # Add categories to category subgraph
        try:
            added_categories = self._add_categories(categories, filters)
            logger.info(f"Successfully added {len(added_categories)} categories to subgraph")
        except Exception as e:
            logger.error(f"Error adding categories to subgraph: {e}")
            import traceback
            traceback.print_exc()
            added_categories = []

        return {
            "deleted_entities": deleted_entities, 
            "added_entities": added_entities,
            "added_categories": added_categories
        }

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data using dual-path retrieval.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "entity_results": List of results from entity subgraph search.
                - "category_results": List of results from category subgraph search.
        """
        # Path 1: Category-aware retrieval (context-aware)
        category_results = self._search_category_subgraph(query, filters, limit)
        
        # Path 2: Traditional entity retrieval (embedding-based)
        entity_results = self._search_entity_subgraph(query, filters, limit)
        
        return {
            "entity_results": entity_results,
            "category_results": category_results
        }

    def delete_all(self, filters):
        """Delete all nodes and relationships for a user or specific agent."""
        if filters.get("agent_id"):
            # Delete from memzero
            cypher_memzero = """
            MATCH (n:Entity {user_id: $user_id, agent_id: $agent_id})
            DETACH DELETE n
            """
            # --- NEW: Delete from memzero_category (Category nodes) ---
            cypher_category = """
            MATCH (n:Category {user_id: $user_id, agent_id: $agent_id})
            DETACH DELETE n
            """
            params = {"user_id": filters["user_id"], "agent_id": filters["agent_id"]}
            self.graph.query(cypher_memzero, params=params)
            self.graph.query(cypher_category, params=params) # Execute category deletion
        else:
            # Delete from memzero
            cypher_memzero = """
            MATCH (n:Entity {user_id: $user_id})
            DETACH DELETE n
            """
            # --- NEW: Delete from memzero_category (Category nodes) ---
            cypher_category = """
            MATCH (n:Category {user_id: $user_id})
            DETACH DELETE n
            """
            params = {"user_id": filters["user_id"]}
            self.graph.query(cypher_memzero, params=params)
            self.graph.query(cypher_category, params=params) # Execute category deletion

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.

        Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
                Supports 'user_id' (required) and 'agent_id' (optional).
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'source': The source node name.
                - 'relationship': The relationship type.
                - 'target': The target node name.
        """
        # Build query based on whether agent_id is provided
        # This function should only query the main memzero graph for the complete picture
        if filters.get("agent_id"):
            query = """
            MATCH (n:Entity {user_id: $user_id, agent_id: $agent_id})-[r]->(m:Entity {user_id: $user_id, agent_id: $agent_id})
            RETURN n.name AS source, type(r) AS relationship, m.name AS target
            LIMIT $limit
            """
            params = {"user_id": filters["user_id"], "agent_id": filters["agent_id"], "limit": limit}
        else:
            query = """
            MATCH (n:Entity {user_id: $user_id})-[r]->(m:Entity {user_id: $user_id})
            RETURN n.name AS source, type(r) AS relationship, m.name AS target
            LIMIT $limit
            """
            params = {"user_id": filters["user_id"], "limit": limit}

        results = self.graph.query(query, params=params)

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        
        prompt = f"""You are a smart assistant focused on extracting entities and their types from user input.
                        Your primary goal is to identify and list all relevant entities present in the text.

                        **Entity Definitions:**
                        *   **Core Entities:** These are specific, identifiable things like people, places, organizations, products, dates, times, etc.
                        *   **Self-Reference Entity:** If the user message contains first-person pronouns such as 'I', 'me', 'my', 'mine', or 'myself', you **MUST** treat the provided `user_id` as a specific entity. The entity name will be the `user_id` itself, and its `entity_type` will be 'user'.
                        *   **Other Entities:** Identify other common entities like locations, organizations, products, dates, etc., and assign an appropriate `entity_type`.

                        **Important Rules:**
                        1.  **NEVER answer the user's question directly.** Your sole purpose is to extract entities.
                        2.  **Always map first-person pronouns to `user_id`**. This is a mandatory entity extraction.
                        3.  If an entity is ambiguous or its type cannot be determined, **do not include it** in the output.

                        ---

                        **Examples:**

                        **Example 1: Simple Query with Self-Reference**
                        **Input Data:** "Where can I find good supermarkets?"
                        **User ID:** "test_user_memory_manager_optimized"

                        **Expected Output (Tool Call):**
                        ```json
                        {{
                        "entities": [
                            {{"entity": "test_user_memory_manager_optimized", "entity_type": "user"}},
                            {{"entity": "supermarkets", "entity_type": "place_type"}}
                        ]
                        }}
                        ```"""
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {"role": "user", "content": json.dumps({"user_id": filters["user_id"], "data": data})},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.debug(f"Entity type map: {entity_type_map}\n search_results={search_results}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Eshtablish relations among the extracted nodes."""
        if self.config.graph_store.custom_prompt:
            messages = [
                {
                    "role": "system",
                    "content": EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters["user_id"]).replace(
                        "CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}"
                    ),
                },
                {"role": "user", "content": data},
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters["user_id"]),
                },
                {
                    "role": "user",
                    "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}",
                },
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities["tool_calls"]:
            entities = extracted_entities["tool_calls"][0]["arguments"]["entities"]

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes among and their respective incoming and outgoing relations in memzero."""
        # 此函数用于`add`方法中查找主memzero图中的现有相似节点，不应为新的搜索逻辑修改。
        result_relations = []

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)

            # Build query based on whether agent_id is provided
            if filters.get("agent_id"):
                cypher_query = """
                CALL vector_search.search("memzero", $limit, $n_embedding)
                YIELD distance, node, similarity
                WITH node AS n, similarity
                WHERE n:Entity AND n.user_id = $user_id AND n.agent_id = $agent_id AND n.embedding IS NOT NULL AND similarity >= $threshold
                MATCH (n)-[r]->(m:Entity)
                RETURN n.name AS source, id(n) AS source_id, type(r) AS relationship, id(r) AS relation_id, m.name AS destination, id(m) AS destination_id, similarity
                UNION
                CALL vector_search.search("memzero", $limit, $n_embedding)
                YIELD distance, node, similarity
                WITH node AS n, similarity
                WHERE n:Entity AND n.user_id = $user_id AND n.agent_id = $agent_id AND n.embedding IS NOT NULL AND similarity >= $threshold
                MATCH (m:Entity)-[r]->(n)
                RETURN m.name AS source, id(m) AS source_id, type(r) AS relationship, id(r) AS relation_id, n.name AS destination, id(n) AS destination_id, similarity
                ORDER BY similarity DESC
                LIMIT $limit;
                """
                params = {
                    "n_embedding": n_embedding,
                    "threshold": self.threshold,
                    "user_id": filters["user_id"],
                    "agent_id": filters["agent_id"],
                    "limit": limit,
                }
            else:
                cypher_query = """
                CALL vector_search.search("memzero", $limit, $n_embedding)
                YIELD distance, node, similarity
                WITH node AS n, similarity
                WHERE n:Entity AND n.user_id = $user_id AND n.embedding IS NOT NULL AND similarity >= $threshold
                MATCH (n)-[r]->(m:Entity)
                RETURN n.name AS source, id(n) AS source_id, type(r) AS relationship, id(r) AS relation_id, m.name AS destination, id(m) AS destination_id, similarity
                UNION
                CALL vector_search.search("memzero", $limit, $n_embedding)
                YIELD distance, node, similarity
                WITH node AS n, similarity
                WHERE n:Entity AND n.user_id = $user_id AND n.embedding IS NOT NULL AND similarity >= $threshold
                MATCH (m:Entity)-[r]->(n)
                RETURN m.name AS source, id(m) AS source_id, type(r) AS relationship, id(r) AS relation_id, n.name AS destination, id(n) AS destination_id, similarity
                ORDER BY similarity DESC
                LIMIT $limit;
                """
                params = {
                    "n_embedding": n_embedding,
                    "threshold": self.threshold,
                    "user_id": filters["user_id"],
                    "limit": limit,
                }

            ans = self.graph.query(cypher_query, params=params)
            result_relations.extend(ans)

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)
        system_prompt, user_prompt = get_delete_messages(search_output_string, data, filters["user_id"])

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )
        to_be_deleted = []
        for item in memory_updates["tool_calls"]:
            if item["name"] == "delete_graph_memory":
                to_be_deleted.append(item["arguments"])
        # in case if it is not in the correct format
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Build the agent filter for the query
            agent_filter = ""
            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }

            if agent_id:
                agent_filter = "AND n.agent_id = $agent_id AND m.agent_id = $agent_id"
                params["agent_id"] = agent_id

            # Delete the specific relationship between nodes in memzero
            cypher = f"""
            MATCH (n:Entity {{name: $source_name, user_id: $user_id}})
            -[r:{relationship}]->
            (m:Entity {{name: $dest_name, user_id: $user_id}})
            WHERE 1=1 {agent_filter}
            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """

            result = self.graph.query(cypher, params=params)
            results.append(result)


        return results

    # added Entity label to all nodes for vector search to work
    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        results = []

        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # types
            source_type = entity_type_map.get(source, "__User__")
            destination_type = entity_type_map.get(destination, "__User__")

            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # search for the nodes with the closest embeddings
            source_node_search_result = self._search_source_node(source_embedding, filters, threshold=0.9)
            destination_node_search_result = self._search_destination_node(dest_embedding, filters, threshold=0.9)

            # Prepare agent_id for node creation
            agent_id_clause = ""
            if agent_id:
                agent_id_clause = ", agent_id: $agent_id"

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                cypher = f"""
                    MATCH (source:Entity)
                    WHERE id(source) = $source_id
                    MERGE (destination:{destination_type}:Entity {{name: $destination_name, user_id: $user_id{agent_id_clause}}})
                    ON CREATE SET
                        destination.created = timestamp(),
                        destination.embedding = $destination_embedding,
                        destination:Entity
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created = timestamp()
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """

                params = {
                    "source_id": source_node_search_result[0]["id(source_candidate)"],
                    "destination_name": destination,
                    "destination_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id

            elif destination_node_search_result and not source_node_search_result:
                cypher = f"""
                    MATCH (destination:Entity)
                    WHERE id(destination) = $destination_id
                    MERGE (source:{source_type}:Entity {{name: $source_name, user_id: $user_id{agent_id_clause}}})
                    ON CREATE SET
                        source.created = timestamp(),
                        source.embedding = $source_embedding,
                        source:Entity
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created = timestamp()
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """

                params = {
                    "destination_id": destination_node_search_result[0]["id(destination_candidate)"],
                    "source_name": source,
                    "source_embedding": source_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id

            elif source_node_search_result and destination_node_search_result:
                cypher = f"""
                    MATCH (source:Entity)
                    WHERE id(source) = $source_id
                    MATCH (destination:Entity)
                    WHERE id(destination) = $destination_id
                    MERGE (source)-[r:{relationship}]->(destination)
                    ON CREATE SET 
                        r.created_at = timestamp(),
                        r.updated_at = timestamp()
                    RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                    """
                params = {
                    "source_id": source_node_search_result[0]["id(source_candidate)"],
                    "destination_id": destination_node_search_result[0]["id(destination_candidate)"],
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id

            else:
                cypher = f"""
                    MERGE (n:{source_type}:Entity {{name: $source_name, user_id: $user_id{agent_id_clause}}})
                    ON CREATE SET n.created = timestamp(), n.embedding = $source_embedding, n:Entity
                    ON MATCH SET n.embedding = $source_embedding
                    MERGE (m:{destination_type}:Entity {{name: $dest_name, user_id: $user_id{agent_id_clause}}})
                    ON CREATE SET m.created = timestamp(), m.embedding = $dest_embedding, m:Entity
                    ON MATCH SET m.embedding = $dest_embedding
                    MERGE (n)-[rel:{relationship}]->(m)
                    ON CREATE SET rel.created = timestamp()
                    RETURN n.name AS source, type(rel) AS relationship, m.name AS target
                    """
                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id

            result = self.graph.query(cypher, params=params)
            results.append(result)

            # --- NEW: 同步更新memzero_category ---
            # 如果源是用户，且目标是非用户实体（一级类目）
            if source == user_id and destination_type != "user":
                self._add_to_memzero_category(user_id, agent_id, destination, dest_embedding)
            # 如果目标是用户，且源是非用户实体（一级类目）
            elif destination == user_id and source_type != "user":
                self._add_to_memzero_category(user_id, agent_id, source, source_embedding)

        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            # Use the sanitization function for relationships to handle special characters
            item["relationship"] = sanitize_relationship_for_cypher(item["relationship"].lower().replace(" ", "_"))
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, threshold=0.9):
        """Search for source nodes with similar embeddings."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)

        if agent_id:
            cypher = """
                CALL vector_search.search("memzero", 1, $source_embedding) 
                YIELD distance, node, similarity
                WITH node AS source_candidate, similarity
                WHERE source_candidate.user_id = $user_id 
                AND source_candidate.agent_id = $agent_id 
                AND similarity >= $threshold
                RETURN id(source_candidate);
                """
            params = {
                "source_embedding": source_embedding,
                "user_id": user_id,
                "agent_id": agent_id,
                "threshold": threshold,
            }
        else:
            cypher = """
                CALL vector_search.search("memzero", 1, $source_embedding) 
                YIELD distance, node, similarity
                WITH node AS source_candidate, similarity
                WHERE source_candidate.user_id = $user_id 
                AND similarity >= $threshold
                RETURN id(source_candidate);
                """
            params = {
                "source_embedding": source_embedding,
                "user_id": user_id,
                "threshold": threshold,
            }

        result = self.graph.query(cypher, params=params)
        return result

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        """Search for destination nodes with similar embeddings."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)

        if agent_id:
            cypher = """
                CALL vector_search.search("memzero", 1, $destination_embedding) 
                YIELD distance, node, similarity
                WITH node AS destination_candidate, similarity
                WHERE node.user_id = $user_id 
                AND node.agent_id = $agent_id 
                AND similarity >= $threshold
                RETURN id(destination_candidate);
                """
            params = {
                "destination_embedding": destination_embedding,
                "user_id": user_id,
                "agent_id": agent_id,
                "threshold": threshold,
            }
        else:
            cypher = """
                CALL vector_search.search("memzero", 1, $destination_embedding) 
                YIELD distance, node, similarity
                WITH node AS destination_candidate, similarity
                WHERE node.user_id = $user_id 
                AND similarity >= $threshold
                RETURN id(destination_candidate);
                """
            params = {
                "destination_embedding": destination_embedding,
                "user_id": user_id,
                "threshold": threshold,
            }

        result = self.graph.query(cypher, params=params)
        return result

    def _fetch_existing_indexes(self):
        """
        Retrieves information about existing indexes and vector indexes in the Memgraph database.

        Returns:
            dict: A dictionary containing lists of existing indexes and vector indexes.
        """

        index_exists = list(self.graph.query("SHOW INDEX INFO;"))
        vector_index_exists = list(self.graph.query("SHOW VECTOR INDEX INFO;"))
        return {"index_exists": index_exists, "vector_index_exists": vector_index_exists}

    # --- NEW HELPER METHODS FOR CATEGORY SUBGRAPH ---

    def _extract_categories_from_data(self, data, filters):
        """Extract categories from user data using LLM."""
        _tools = [EXTRACT_CATEGORIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_CATEGORIES_STRUCT_TOOL]
        
        # Get existing user categories for context
        existing_categories = self._get_existing_user_categories(filters)
        
        # Prepare input for category extraction
        memo_input = pack_category_input(existing_categories, data)
        
        # Get category extraction prompt
        topic_examples = get_default_profiles()
        prompt = get_category_extraction_prompt(topic_examples)
        
        search_results = self.llm.generate_response(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": memo_input},
            ],
            tools=_tools,
        )

        categories = []
        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_categories":
                    continue
                for item in tool_call["arguments"]["categories"]:
                    categories.append({
                        "topic": item["topic"],
                        "sub_topic": item["sub_topic"], 
                        "memo": item["memo"]
                    })
        except Exception as e:
            logger.exception(f"Error in category extraction: {e}")
            
        logger.debug(f"Extracted categories: {categories}")
        return categories

    def _add_categories(self, categories, filters):
        """Add categories to the category subgraph."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id")
        results = []
        
        for category in categories:
            topic = category["topic"]
            sub_topic = category["sub_topic"]
            memo = category["memo"]
            
            # Create embeddings for each level
            topic_embedding = self.embedding_model.embed(topic)
            sub_topic_embedding = self.embedding_model.embed(sub_topic)
            memo_embedding = self.embedding_model.embed(memo)
            
            # Build category hierarchy: user -> topic -> sub_topic -> memo
            result = self._add_category_hierarchy(
                user_id, agent_id, topic, sub_topic, memo,
                topic_embedding, sub_topic_embedding, memo_embedding
            )
            results.append(result)
            
        return results

    def _add_category_hierarchy(self, user_id, agent_id, topic, sub_topic, memo, 
                               topic_embedding, sub_topic_embedding, memo_embedding):
        """Add the complete category hierarchy to the graph."""
        agent_id_clause = ""
        if agent_id:
            agent_id_clause = ", agent_id: $agent_id"
            
        cypher = f"""
        // Ensure user node exists
        MERGE (user:Entity {{name: $user_id, user_id: $user_id{agent_id_clause}}})
        ON CREATE SET user.created = timestamp()
        
        // Create topic node
        MERGE (topic:Category:Entity {{name: $topic, user_id: $user_id{agent_id_clause}}})
        ON CREATE SET topic.created = timestamp(), topic.embedding = $topic_embedding
        ON MATCH SET topic.embedding = $topic_embedding
        
        // Create sub_topic node  
        MERGE (sub_topic:Category:Entity {{name: $sub_topic, user_id: $user_id{agent_id_clause}}})
        ON CREATE SET sub_topic.created = timestamp(), sub_topic.embedding = $sub_topic_embedding
        ON MATCH SET sub_topic.embedding = $sub_topic_embedding
        
        // Create memo node
        MERGE (memo:Category:Entity {{name: $memo, user_id: $user_id{agent_id_clause}}})
        ON CREATE SET memo.created = timestamp(), memo.embedding = $memo_embedding
        ON MATCH SET memo.embedding = $memo_embedding
        
        // Create relationships: user -> topic -> sub_topic -> memo
        MERGE (user)-[r1:HAS_TOPIC]->(topic)
        ON CREATE SET r1.created = timestamp()
        
        MERGE (topic)-[r2:HAS_SUB_TOPIC]->(sub_topic)
        ON CREATE SET r2.created = timestamp()
        
        MERGE (sub_topic)-[r3:HAS_MEMO]->(memo)
        ON CREATE SET r3.created = timestamp()
        
        RETURN user.name AS user, topic.name AS topic, sub_topic.name AS sub_topic, memo.name AS memo
        """
        
        params = {
            "user_id": user_id,
            "topic": topic,
            "sub_topic": sub_topic,
            "memo": memo,
            "topic_embedding": topic_embedding,
            "sub_topic_embedding": sub_topic_embedding,
            "memo_embedding": memo_embedding,
        }
        if agent_id:
            params["agent_id"] = agent_id
            
        result = self.graph.query(cypher, params=params)
        logger.debug(f"Added category hierarchy: {user_id} -> {topic} -> {sub_topic} -> {memo}")
        return result

    def _search_category_subgraph(self, query, filters, limit=100):
        """Search the category subgraph using context-aware retrieval."""
        # First, classify the query to identify relevant categories
        query_categories = self._classify_query_categories(query, filters)
        
        if not query_categories:
            logger.info("No relevant categories identified for query")
            return []
            
        # Search for matching categories in the subgraph
        results = []
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id")
        
        for category in query_categories:
            topic = category["topic"]
            sub_topic = category["sub_topic"]
            confidence = category["confidence"]
            
            if confidence < 0.3:  # Skip low confidence categories
                continue
                
            # Search for matching category paths
            category_results = self._search_category_paths(user_id, agent_id, topic, sub_topic, limit)
            results.extend(category_results)
            
        return results[:limit]

    def _search_entity_subgraph(self, query, filters, limit=100):
        """Search the entity subgraph using traditional embedding-based retrieval."""
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        
        if not search_output:
            return []
            
        # Format results for consistency
        results = []
        for item in search_output:
            results.append({
                "source": item["source"],
                "relationship": item["relationship"], 
                "destination": item["destination"]
            })
            
        return results[:limit]

    def _classify_query_categories(self, query, filters):
        """Classify query to identify relevant categories."""
        _tools = [CLASSIFY_QUERY_CATEGORIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [CLASSIFY_QUERY_CATEGORIES_STRUCT_TOOL]
            
        prompt = """You are a smart assistant that classifies user queries to identify which user profile categories might be relevant for context-aware retrieval.

Use these exact category names that match the stored data:
- basic_info: name, age, gender, location, contact
- work: title, company, industry, skills, experience  
- education: degree, school, field, graduation_year
- interests: hobbies, sports, music, movies, books, travel
- health: conditions, medications, allergies, fitness
- family: spouse, children, parents, siblings
- social: friends, social_media, groups, activities

For example:
- "Tell me some swimming clubs near" -> might be relevant to "basic_info" with "location" subcategory and "interests" with "hobbies" subcategory
- "What movies do I like?" -> relevant to "interests" category with "movies" subcategory
- "Where do I work?" -> relevant to "work" category
- "How old is Tom?" -> relevant to "basic_info" with "age" subcategory
- "What school does Tom go to?" -> relevant to "education" with "school" subcategory

Analyze the query and identify potential categories that might contain relevant information using the exact category names above."""

        search_results = self.llm.generate_response(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            tools=_tools,
        )

        categories = []
        try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "classify_query_categories":
                    continue
                for item in tool_call["arguments"]["categories"]:
                    categories.append({
                        "topic": item["topic"],
                        "sub_topic": item["sub_topic"],
                        "confidence": item["confidence"]
                    })
        except Exception as e:
            logger.exception(f"Error in query classification: {e}")
            
        return categories

    def _search_category_paths(self, user_id, agent_id, topic, sub_topic, limit):
        """Search for specific category paths in the subgraph."""
        agent_filter = ""
        params = {"user_id": user_id, "topic": topic, "sub_topic": sub_topic, "limit": limit}
        
        if agent_id:
            agent_filter = "AND user.agent_id = $agent_id AND topic.agent_id = $agent_id AND sub_topic.agent_id = $agent_id AND memo.agent_id = $agent_id"
            params["agent_id"] = agent_id
            
        cypher = f"""
        MATCH (user:Entity {{user_id: $user_id}})-[:HAS_TOPIC]->(topic:Category {{name: $topic}})
        -[:HAS_SUB_TOPIC]->(sub_topic:Category {{name: $sub_topic}})
        -[:HAS_MEMO]->(memo:Category)
        WHERE 1=1 {agent_filter}
        RETURN user.name AS source, 'HAS_TOPIC' AS relationship, topic.name AS destination
        UNION
        MATCH (user:Entity {{user_id: $user_id}})-[:HAS_TOPIC]->(topic:Category {{name: $topic}})
        -[:HAS_SUB_TOPIC]->(sub_topic:Category {{name: $sub_topic}})
        -[:HAS_MEMO]->(memo:Category)
        WHERE 1=1 {agent_filter}
        RETURN topic.name AS source, 'HAS_SUB_TOPIC' AS relationship, sub_topic.name AS destination
        UNION
        MATCH (user:Entity {{user_id: $user_id}})-[:HAS_TOPIC]->(topic:Category {{name: $topic}})
        -[:HAS_SUB_TOPIC]->(sub_topic:Category {{name: $sub_topic}})
        -[:HAS_MEMO]->(memo:Category)
        WHERE 1=1 {agent_filter}
        RETURN sub_topic.name AS source, 'HAS_MEMO' AS relationship, memo.name AS destination
        LIMIT $limit
        """
        
        results = self.graph.query(cypher, params=params)
        return [dict(item) for item in results]

    def _get_existing_user_categories(self, filters):
        """Get existing user categories for context in extraction."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id")
        
        agent_filter = ""
        params = {"user_id": user_id}
        if agent_id:
            agent_filter = "AND user.agent_id = $agent_id AND topic.agent_id = $agent_id AND sub_topic.agent_id = $agent_id AND memo.agent_id = $agent_id"
            params["agent_id"] = agent_id
            
        cypher = f"""
        MATCH (user:Entity {{user_id: $user_id}})-[:HAS_TOPIC]->(topic:Category)
        -[:HAS_SUB_TOPIC]->(sub_topic:Category)
        -[:HAS_MEMO]->(memo:Category)
        WHERE 1=1 {agent_filter}
        RETURN topic.name AS topic, sub_topic.name AS sub_topic, memo.name AS memo
        LIMIT 50
        """
        
        results = self.graph.query(cypher, params=params)
        
        # Format as string for input
        if not results:
            return "No existing categories found."
            
        category_lines = []
        for item in results:
            category_lines.append(f"- {item['topic']}\t{item['sub_topic']}\t{item['memo']}")
            
        return "\n".join(category_lines)

    def _add_to_memzero_category(self, user_id, agent_id, category_name, category_embedding):
        """Adds a user-category relationship to the memzero_category index."""
        agent_id_clause = ""
        if agent_id:
            agent_id_clause = ", agent_id: $agent_id"

        # 确保用户节点作为Entity存在（应该已经从memzero逻辑中存在）
        # 合并带有 :Category 标签的类目节点 (也带有 :Entity 标签以保持与其他节点的一致性)
        # 在 memzero_category 中创建从用户到类目的关系
        cypher = f"""
        MERGE (user:Entity {{name: $user_id, user_id: $user_id{agent_id_clause}}})
        ON CREATE SET user.created = timestamp()
        MERGE (category:Category:Entity {{name: $category_name, user_id: $user_id{agent_id_clause}}})
        ON CREATE SET category.created = timestamp(), category.embedding = $category_embedding
        ON MATCH SET category.embedding = $category_embedding // 如果匹配到，更新嵌入
        MERGE (user)-[r:HAS_CATEGORY]->(category)
        ON CREATE SET r.created = timestamp()
        RETURN user.name AS source, type(r) AS relationship, category.name AS target
        """
        params = {
            "user_id": user_id,
            "category_name": category_name,
            "category_embedding": category_embedding,
        }
        if agent_id:
            params["agent_id"] = agent_id
        self.graph.query(cypher, params=params)
        logger.debug(f"已为用户 '{user_id}' 在 memzero_category 中添加/更新 '{category_name}'")
