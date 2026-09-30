from retrieval.models import RetrievalChunk


class ContextBuilder:

    def build(self, chunks: list[RetrievalChunk]) -> str:
        if not chunks:
            return ""

        context_parts = []

        for chunk in chunks:
            source_name = chunk.source.split("\\")[-1].split("/")[-1]

            page = (
                str(chunk.page_number)
                if chunk.page_number is not None
                else "Unknown"
            )

            header = (
                f"[Source: {source_name} | "
                f"Page: {page} | "
                f"Type: {chunk.element_type} | "
                f"Chunk ID: {chunk.chunk_id}]"
            )

            context_parts.append(
                f"{header}\n\n{chunk.content}"
            )

        return "\n\n---\n\n".join(context_parts)