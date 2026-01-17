from typing import Dict, List
from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .llm_service import LLMService

class WorkflowExecutor:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
    
    def build_execution_order(self, nodes: List[Dict], edges: List[Dict]) -> List[str]:
        """Build execution order from nodes and edges"""
        # Create adjacency list
        graph = {node['id']: [] for node in nodes}
        for edge in edges:
            graph[edge['source']].append(edge['target'])
        
        # Find start node (userQuery)
        start_node = next((n['id'] for n in nodes if n['data']['componentType'] == 'userQuery'), None)
        
        if not start_node:
            raise ValueError("No User Query component found")
        
        # Simple topological order (DFS)
        order = []
        visited = set()
        
        def dfs(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            order.append(node_id)
            for neighbor in graph.get(node_id, []):
                dfs(neighbor)
        
        dfs(start_node)
        return order
    
    async def execute_workflow(
        self,
        query: str,
        nodes: List[Dict],
        edges: List[Dict],
        node_configs: Dict
    ) -> Dict:
        """Execute the workflow"""
        
        execution_order = self.build_execution_order(nodes, edges)
        
        # Track execution state
        state = {
            "query": query,
            "context": None,
            "response": None
        }
        
        for node_id in execution_order:
            node = next(n for n in nodes if n['id'] == node_id)
            component_type = node['data']['componentType']
            config = node_configs.get(node_id, {})
            
            if component_type == 'userQuery':
                # Already have the query
                pass
            
            elif component_type == 'knowledgeBase':
                # Retrieve context from vector store
                collection_name = config.get('collectionName', 'default')
                embedding_model = config.get('embeddingModel', 'openai')
                
                try:
                    results = self.embedding_service.query_collection(
                        collection_name=collection_name,
                        query_text=state["query"],
                        embedding_model=embedding_model,
                        n_results=3
                    )
                    state["context"] = "\n\n".join([r["content"] for r in results])
                except:
                    state["context"] = None
            
            elif component_type == 'llmEngine':
                # Generate response using LLM
                llm_response = await self.llm_service.generate_response(
                    query=state["query"],
                    context=state.get("context"),
                    system_prompt=config.get('systemPrompt'),
                    provider=config.get('provider', 'openai'),
                    model=config.get('model', 'gpt-4'),
                    temperature=config.get('temperature', 0.7),
                    enable_web_search=config.get('enableWebSearch', False),
                    search_provider=config.get('searchProvider', 'serpapi')
                )
                state["response"] = llm_response["response"]
                state["metadata"] = {
                    "model": llm_response["model"],
                    "web_context": llm_response.get("web_context")
                }
            
            elif component_type == 'output':
                # Format final output
                pass
        
        return state