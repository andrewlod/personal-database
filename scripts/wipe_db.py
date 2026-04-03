"""
Script to wipe all data from the Weaviate vector database.
Requires explicit confirmation to prevent accidental deletions.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.embedder import WeaviateVectorDB

logger = logging.getLogger(__name__)


def wipe_database(
    host: str = "localhost",
    port: int = 8080,
    class_name: str = "PersonalKnowledge",
    yes: bool = False,
) -> bool:
    """
    Delete all objects from the Weaviate collection.

    Args:
        host: Weaviate host
        port: Weaviate port
        class_name: Collection name to wipe
        yes: Skip confirmation prompt

    Returns:
        True if successful, False otherwise
    """
    if not yes:
        print(
            f"WARNING: This will delete ALL objects from the '{class_name}' collection."
        )
        print("This action cannot be undone.")
        response = input("Type 'yes' to confirm: ")
        if response.strip().lower() != "yes":
            print("Aborted.")
            return False

    try:
        import weaviate
    except ImportError:
        logger.error("weaviate-client package not installed")
        return False

    try:
        client = weaviate.connect_to_local(host=host, port=port, grpc_port=50051)

        if not client.is_ready():
            logger.error("Weaviate is not ready")
            return False

        if not client.collections.exists(class_name):
            print(f"Collection '{class_name}' does not exist. Nothing to wipe.")
            client.close()
            return True

        collection = client.collections.get(class_name)

        # Get count before deletion
        count = len(collection)
        print(f"Found {count} objects in '{class_name}' collection.")

        if count == 0:
            print("Collection is already empty.")
            client.close()
            return True

        # Delete and recreate the collection (simplest way to wipe all data in Weaviate v4)
        client.collections.delete(class_name)

        # Recreate the collection with the same schema
        from weaviate.classes.config import Property, DataType

        client.collections.create(
            name=class_name,
            description="Personal knowledge base documents",
            vectorizer_config=None,
            properties=[
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="The text content of the chunk",
                ),
                Property(
                    name="document_id",
                    data_type=DataType.TEXT,
                    description="ID of the source document",
                ),
                Property(
                    name="chunk_id",
                    data_type=DataType.TEXT,
                    description="ID of this specific chunk",
                ),
                Property(
                    name="chunk_index",
                    data_type=DataType.INT,
                    description="Index of this chunk within the document",
                ),
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="Title of the source document",
                ),
                Property(
                    name="source_url",
                    data_type=DataType.TEXT,
                    description="URL of the source document (if applicable)",
                ),
                Property(
                    name="timestamp",
                    data_type=DataType.NUMBER,
                    description="Timestamp when the document was processed",
                ),
                Property(
                    name="word_count",
                    data_type=DataType.INT,
                    description="Number of words in the chunk",
                ),
                Property(
                    name="metadata_json",
                    data_type=DataType.TEXT,
                    description="Additional metadata as JSON string",
                ),
            ],
        )

        print(
            f"Successfully deleted {count} objects and recreated '{class_name}' collection."
        )

        client.close()
        return True

        # Delete all objects
        from weaviate.classes.query import Filter

        collection.data.delete_many(
            where=Filter.by_property("_creationTimeUnix").greater_or_equal("0")
        )

        # Verify deletion
        remaining = len(collection)
        if remaining == 0:
            print(f"Successfully deleted {count} objects from '{class_name}'.")
        else:
            print(f"WARNING: {remaining} objects remain after deletion (expected 0).")

        client.close()
        return True

    except Exception as e:
        logger.error(f"Failed to wipe database: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Wipe all data from the Weaviate vector database"
    )
    parser.add_argument("--host", default="localhost", help="Weaviate host")
    parser.add_argument("--port", type=int, default=8080, help="Weaviate port")
    parser.add_argument(
        "--class-name", default="PersonalKnowledge", help="Collection name to wipe"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    success = wipe_database(
        host=args.host, port=args.port, class_name=args.class_name, yes=args.yes
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
