"""
Natural language query interface for GraphRAG system.
Combines vector similarity search with graph traversal for complex multi-hop queries.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from src.config.settings import get_settings
from src.database.connection import Neo4jConnection
from src.rag.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class GraphRAGQuery:
    """Natural language query interface for the GraphRAG knowledge system."""

    def __init__(self, connection: Neo4jConnection) -> None:
        """
        Initialize query interface.

        Args:
            connection: Neo4j connection instance
        """
        self.connection = connection
        self.settings = get_settings()
        self.embedding_manager = EmbeddingManager(connection)

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.settings.openai.model,
            api_key=self.settings.openai.api_key,
            temperature=0.3,  # Slightly creative for better answers
        )

    def vector_search(
        self, query: str, top_k: int = 5, score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            query: Natural language query
            top_k: Number of results to retrieve
            score_threshold: Minimum similarity score

        Returns:
            List of relevant chunks with metadata
        """
        logger.info(f"Performing vector search for: {query}")
        results = self.embedding_manager.similarity_search(
            query, top_k=top_k, score_threshold=score_threshold
        )
        return results

    def graph_context_expansion(
        self, chunk_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Expand context using graph relationships.

        Args:
            chunk_ids: List of chunk IDs from vector search

        Returns:
            Dictionary with related entities and relationships
        """
        logger.info("Expanding context using graph relationships...")

        query = """
        MATCH (c:Chunk)
        WHERE c.chunk_id IN $chunk_ids
        MATCH (c)<-[:HAS_CHUNK]-(d:Document)
        OPTIONAL MATCH (c)-[:MENTIONS_PERSON]->(p:Person)
        OPTIONAL MATCH (c)-[:MENTIONS_PROCEDURE]->(pr:Procedure)
        OPTIONAL MATCH (d)-[:GENERATED_FOR]->(proj:Project)
        OPTIONAL MATCH (person:Person)-[:MANAGES]->(proj)

        RETURN
            d.file_name AS document,
            d.type AS document_type,
            collect(DISTINCT p.name) AS mentioned_persons,
            collect(DISTINCT pr.id) AS mentioned_procedures,
            collect(DISTINCT proj.name) AS related_projects,
            collect(DISTINCT person.name) AS project_managers
        """

        results = self.connection.execute_query(query, {"chunk_ids": chunk_ids})

        # Aggregate context
        context = {
            "documents": [],
            "persons": set(),
            "procedures": set(),
            "projects": set(),
            "managers": set(),
        }

        for result in results:
            context["documents"].append(
                {
                    "name": result.get("document", ""),
                    "type": result.get("document_type", ""),
                }
            )
            context["persons"].update(
                [p for p in result.get("mentioned_persons", []) if p]
            )
            context["procedures"].update(
                [pr for pr in result.get("mentioned_procedures", []) if pr]
            )
            context["projects"].update(
                [proj for proj in result.get("related_projects", []) if proj]
            )
            context["managers"].update(
                [m for m in result.get("project_managers", []) if m]
            )

        # Convert sets to lists
        return {
            "documents": context["documents"],
            "persons": list(context["persons"]),
            "procedures": list(context["procedures"]),
            "projects": list(context["projects"]),
            "managers": list(context["managers"]),
        }

    def execute_cypher_query(self, cypher: str) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query.

        Args:
            cypher: Cypher query string

        Returns:
            Query results
        """
        logger.info(f"Executing custom Cypher query: {cypher[:100]}...")
        return self.connection.execute_query(cypher)

    def hybrid_query(
        self, natural_language_query: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Execute hybrid query combining vector search and graph traversal.

        Args:
            natural_language_query: User's question in natural language
            top_k: Number of chunks to retrieve

        Returns:
            Query results with answer and supporting evidence
        """
        logger.info(f"Processing hybrid query: {natural_language_query}")

        # Step 1: Vector similarity search
        similar_chunks = self.vector_search(
            natural_language_query, top_k=top_k, score_threshold=0.6
        )

        if not similar_chunks:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base for your query.",
                "chunks": [],
                "context": {},
                "confidence": "low",
            }

        # Step 2: Expand context using graph
        chunk_ids = [chunk["chunk_id"] for chunk in similar_chunks]
        graph_context = self.graph_context_expansion(chunk_ids)

        # Step 3: Generate answer using LLM
        answer = self._generate_answer(
            natural_language_query, similar_chunks, graph_context
        )

        return {
            "answer": answer,
            "chunks": similar_chunks,
            "context": graph_context,
            "confidence": "high" if len(similar_chunks) >= 3 else "medium",
        }

    def _generate_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        graph_context: Dict[str, Any],
    ) -> str:
        """
        Generate final answer using LLM with retrieved context.

        Args:
            question: User's question
            chunks: Retrieved text chunks
            graph_context: Graph-based context

        Returns:
            Generated answer
        """
        # Format context for LLM
        chunk_texts = "\n\n".join(
            [
                f"[Document: {chunk['document_name']} ({chunk['document_type']}), Score: {chunk['score']:.2f}]\n{chunk['text']}"
                for chunk in chunks
            ]
        )

        graph_info = f"""
**Related Information from Knowledge Graph:**
- Documents: {', '.join([d['name'] for d in graph_context['documents'][:5]])}
- Procedures mentioned: {', '.join(graph_context['procedures'][:10])}
- People mentioned: {', '.join(graph_context['persons'][:10])}
- Related projects: {', '.join(graph_context['projects'][:10])}
- Project managers: {', '.join(graph_context['managers'][:10])}
"""

        # Create prompt
        system_message = """You are an expert assistant for a project document knowledge system.
Your role is to answer questions based on the provided context from project documents and the knowledge graph.

**Instructions:**
1. Answer the question directly and concisely
2. Base your answer ONLY on the provided context
3. If the context doesn't contain enough information, say so clearly
4. Cite specific documents when making claims
5. If procedures or standards are mentioned, reference them by their IDs
6. Summarize key risks or issues when relevant
7. Be precise about relationships (who manages what, which documents mention what)
"""

        user_message = f"""**Question:** {question}

**Relevant Document Excerpts:**
{chunk_texts}

{graph_info}

**Task:** Answer the question based on the context above. Be specific and cite sources."""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        # Generate answer
        response = self.llm.invoke(messages)
        return response.content

    def answer_question(self, question: str, top_k: int = 5) -> str:
        """
        Answer a natural language question (simplified interface).

        Args:
            question: User's question
            top_k: Number of chunks to retrieve

        Returns:
            Answer string
        """
        result = self.hybrid_query(question, top_k=top_k)
        return result["answer"]

    def get_project_summary(self, project_name: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary for a specific project.

        Args:
            project_name: Name of the project

        Returns:
            Project summary with all related information
        """
        query = """
        MATCH (proj:Project)
        WHERE proj.name CONTAINS $project_name OR proj.id CONTAINS $project_name
        OPTIONAL MATCH (person:Person)-[:MANAGES]->(proj)
        OPTIONAL MATCH (proj)<-[:GENERATED_FOR]-(d:Document)
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS_PROCEDURE]->(pr:Procedure)
        OPTIONAL MATCH (team:Team)<-[:BELONGS_TO]-(person)

        RETURN
            proj.name AS project_name,
            proj.status AS status,
            collect(DISTINCT person.name) AS managers,
            collect(DISTINCT team.name) AS teams,
            collect(DISTINCT d.file_name) AS documents,
            collect(DISTINCT d.type) AS document_types,
            collect(DISTINCT pr.id) AS procedures
        """

        results = self.connection.execute_query(query, {"project_name": project_name})

        if not results:
            return {"error": f"Project '{project_name}' not found"}

        return results[0]
