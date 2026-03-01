from __future__ import annotations

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import os

from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Neo4jStatus:
    connected: bool
    message: str


class Neo4jTool:
    """
    Enterprise-ready Neo4j connector for Workforce Intelligence System.
    Handles:
    - Safe connection
    - Graph upsert (nodes + relationships)
    - Arbitrary Cypher execution
    """

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "")
        self.enable = os.getenv("ENABLE_NEO4J", "false").lower() == "true"

        self.driver = None

    # ------------------------------------------------------------------
    # Connection Handling
    # ------------------------------------------------------------------
    def connect(self) -> Neo4jStatus:
        if not self.enable:
            return Neo4jStatus(False, "Neo4j disabled via ENABLE_NEO4J=false")

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            return Neo4jStatus(True, "Connected to Neo4j successfully.")
        except Exception as e:
            return Neo4jStatus(False, f"Neo4j connection failed: {str(e)}")

    # ------------------------------------------------------------------
    # Close connection
    # ------------------------------------------------------------------
    def close(self):
        if self.driver:
            self.driver.close()

    # ------------------------------------------------------------------
    # Run arbitrary Cypher query
    # ------------------------------------------------------------------
    def run_query(self, query: str, params: Dict[str, Any] | None = None):
        if not self.driver:
            raise RuntimeError("Neo4j driver not connected. Call connect() first.")

        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record for record in result]

    # ------------------------------------------------------------------
    # Upsert Graph
    # nodes: List[(node_id, label_type, properties)]
    # edges: List[(from_id, rel_type, to_id, properties)]
    # ------------------------------------------------------------------
    def upsert_graph(
        self,
        nodes: List[Tuple[str, str, Dict[str, Any]]],
        edges: List[Tuple[str, str, str, Dict[str, Any]]],
    ) -> str:
        if not self.driver:
            return "Neo4j driver not connected."

        try:
            with self.driver.session() as session:
                # -------------------------
                # Create / Merge Nodes
                # -------------------------
                for node_id, label_type, props in nodes:
                    cypher = f"""
                    MERGE (n:Entity {{id: $id}})
                    SET n.type = $type,
                        n += $props
                    """
                    session.run(
                        cypher,
                        id=node_id,
                        type=label_type,
                        props=props,
                    )

                # -------------------------
                # Create / Merge Relationships
                # -------------------------
                for from_id, rel_type, to_id, props in edges:
                    cypher = """
                    MATCH (a:Entity {id: $from_id})
                    MATCH (b:Entity {id: $to_id})
                    MERGE (a)-[r:REL {type: $rel_type}]->(b)
                    SET r += $props
                    """
                    session.run(
                        cypher,
                        from_id=from_id,
                        to_id=to_id,
                        rel_type=rel_type,
                        props=props,
                    )

            return f"Upserted {len(nodes)} nodes and {len(edges)} edges into Neo4j."

        except Exception as e:
            return f"Neo4j upsert failed: {str(e)}"