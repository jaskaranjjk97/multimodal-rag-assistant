from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from retrieval.models import RetrievalChunk, RetrievalDocument


class RetrievalChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(
        self,
        document: RetrievalDocument,
    ) -> list[RetrievalChunk]:

        if document.element_type == "text":
            return self._chunk_text(document)

        if document.element_type == "table":
            return self._chunk_table(document)

        if document.element_type == "image":
            return self._chunk_image(document)

        raise ValueError(
            f"Unsupported element type: {document.element_type}"
        )

    # ---------------------------------------------------------
    # TEXT CHUNKING
    # ---------------------------------------------------------

    def _chunk_text(
        self,
        document: RetrievalDocument,
    ) -> list[RetrievalChunk]:

        texts = self.text_splitter.split_text(
            document.content
        )

        return [
            self._create_chunk(
                document=document,
                content=text,
            )
            for text in texts
        ]

    # ---------------------------------------------------------
    # TABLE CHUNKING
    # ---------------------------------------------------------

    def _chunk_table(
        self,
        document: RetrievalDocument,
    ) -> list[RetrievalChunk]:

        """
        Keep the table together.

        Tables often contain relationships between columns,
        so splitting individual rows can remove useful context.
        """

        return [
            self._create_chunk(
                document=document,
                content=document.content,
            )
        ]

    # ---------------------------------------------------------
    # IMAGE CHUNKING
    # ---------------------------------------------------------

    def _chunk_image(
        self,
        document: RetrievalDocument,
    ) -> list[RetrievalChunk]:

        """
        Keep the vision-enriched image information together
        as a coherent retrieval unit.

        This is especially important for charts because questions
        about trends may require multiple data points from the
        same image.
        """

        lines = [
            line.strip()
            for line in document.content.splitlines()
            if line.strip()
        ]

        chunks: list[RetrievalChunk] = []

        current_section: str | None = None
        data_points: list[str] = []
        image_type: str | None = None
        description: str | None = None

        for line in lines:

            # -------------------------------------------------
            # IMAGE TYPE
            # -------------------------------------------------

            if line.startswith("Image Type:"):
                image_type = line
                continue

            # -------------------------------------------------
            # IMAGE DESCRIPTION
            # -------------------------------------------------

            if line.startswith("Description:"):
                description = line
                continue

            # -------------------------------------------------
            # SECTION: DATA POINTS
            # -------------------------------------------------

            if line == "Data Points:":
                current_section = "data_points"
                continue

            # -------------------------------------------------
            # SECTION: ENTITIES / KEYWORDS
            # -------------------------------------------------

            if line == "Entities:":
                current_section = "entities"
                continue

            if line == "Keywords:":
                current_section = "keywords"
                continue

            # -------------------------------------------------
            # DATA POINT
            # -------------------------------------------------

            if current_section == "data_points":
                data_points.append(line)

        # -----------------------------------------------------
        # BUILD ONE COHERENT IMAGE RETRIEVAL CHUNK
        # -----------------------------------------------------

        content_parts: list[str] = []

        if image_type:
            content_parts.append(image_type)

        if description:
            content_parts.append(description)

        if data_points:
            content_parts.append("Data Points:")

            for point in data_points:
                content_parts.append(point)

        if content_parts:
            chunks.append(
                self._create_chunk(
                    document=document,
                    content="\n".join(content_parts),
                )
            )

        return chunks

    # ---------------------------------------------------------
    # CHUNK CREATION
    # ---------------------------------------------------------

    @staticmethod
    def _create_chunk(
        document: RetrievalDocument,
        content: str,
    ) -> RetrievalChunk:

        return RetrievalChunk(
            content=content,
            source=document.source,
            page_number=document.page_number,
            element_type=document.element_type,
            chunk_id=str(uuid4()),
            metadata=document.metadata.copy(),
        )