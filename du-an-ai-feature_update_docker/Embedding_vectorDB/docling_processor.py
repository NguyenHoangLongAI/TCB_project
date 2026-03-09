# Embedding_vectorDB/docling_processor.py
"""
Docling-based Document Processor
High-quality document to Markdown conversion with optimized PDF OCR
"""

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TableStructureOptions,
    TableFormerMode,
    AcceleratorOptions,
    AcceleratorDevice
)
from pathlib import Path
from typing import Optional
import logging
import os
import torch

logger = logging.getLogger(__name__)


class DoclingProcessor:
    """
    Advanced document processor using Docling with optimized PDF OCR

    Features:
    - RapidOCR with torch backend for GPU acceleration
    - Enhanced table structure recognition
    - Automatic device detection (CUDA/CPU)
    - Configurable image resolution for OCR
    """

    def __init__(
            self,
            use_ocr: bool = True,
            ocr_lang: list = None,
            use_gpu: bool = True,
            image_scale: float = 2.0,
            force_full_page_ocr: bool = False
    ):
        """
        Initialize Docling processor with optimized settings

        Args:
            use_ocr: Enable OCR for scanned documents
            ocr_lang: OCR languages (default: ['vi', 'en'])
            use_gpu: Use GPU acceleration if available
            image_scale: Image resolution scale for OCR (default: 2.0 = 216 DPI)
            force_full_page_ocr: Force OCR on all pages (slower but more accurate)
        """
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang or ['vi', 'en']
        self.image_scale = image_scale
        self.force_full_page_ocr = force_full_page_ocr

        # Detect GPU availability
        self.has_gpu = torch.cuda.is_available() and use_gpu

        # Configure accelerator options
        accelerator_options = AcceleratorOptions(
            num_threads=8,  # Number of threads for CPU operations
            device=(AcceleratorDevice.CUDA if self.has_gpu
                    else AcceleratorDevice.CPU)
        )

        if self.has_gpu:
            logger.info("🚀 GPU acceleration enabled (CUDA)")
        else:
            logger.info("💻 Running on CPU mode")

        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()

        # Enable OCR with RapidOCR (GPU-accelerated by default)
        if self.use_ocr:
            pipeline_options.do_ocr = True

            # RapidOCR options (simple config - GPU support automatic)
            # Note: RapidOCR will use GPU if available automatically
            pipeline_options.ocr_options = RapidOcrOptions(
                force_full_page_ocr=force_full_page_ocr
            )

            logger.info(
                f"✅ OCR enabled: RapidOCR "
                f"(force_full_page_ocr: {force_full_page_ocr})"
            )

        # Enable table structure recognition with accurate mode
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True,  # Match back to PDF cells
            mode=TableFormerMode.ACCURATE  # Use more accurate model
        )

        # Image generation settings
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = self.image_scale  # Note: images_scale (plural)

        # Enable code recognition
        pipeline_options.do_code_enrichment = True

        # Set accelerator options
        pipeline_options.accelerator_options = accelerator_options

        # Configure format options
        self.format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }

        # Initialize converter
        self.converter = DocumentConverter(
            format_options=self.format_options
        )

        logger.info("✅ Docling processor initialized successfully")
        logger.info(f"   - OCR: {'enabled' if use_ocr else 'disabled'}")
        logger.info(f"   - GPU: {'enabled (CUDA)' if self.has_gpu else 'disabled (CPU)'}")
        logger.info(f"   - Table structure: accurate mode")
        logger.info(f"   - Image scale: {image_scale}x (DPI: {int(108 * image_scale)})")

    def process_document(self, file_path: str) -> Optional[str]:
        """
        Process document to Markdown

        Args:
            file_path: Path to document file

        Returns:
            Markdown content or None if failed
        """
        try:
            logger.info(f"📄 Processing with Docling: {os.path.basename(file_path)}")

            # Convert document
            result = self.converter.convert(file_path)

            # Export to Markdown
            markdown_content = result.document.export_to_markdown()

            # Clean up markdown
            markdown_content = self._clean_markdown(markdown_content)

            logger.info(
                f"✅ Docling conversion successful: {len(markdown_content)} chars"
            )

            return markdown_content

        except Exception as e:
            logger.error(f"❌ Docling processing failed: {e}", exc_info=True)
            return None

    def process_pdf(self, file_path: str) -> Optional[str]:
        """
        Process PDF file specifically

        Args:
            file_path: Path to PDF file

        Returns:
            Markdown content or None if failed
        """
        return self.process_document(file_path)

    def process_docx(self, file_path: str) -> Optional[str]:
        """
        Process DOCX file

        Args:
            file_path: Path to DOCX file

        Returns:
            Markdown content or None if failed
        """
        return self.process_document(file_path)

    def process_image(self, file_path: str) -> Optional[str]:
        """
        Process image file with OCR

        Args:
            file_path: Path to image file

        Returns:
            Markdown content or None if failed
        """
        if not self.use_ocr:
            logger.warning("OCR is disabled, cannot process image")
            return None

        return self.process_document(file_path)

    def _clean_markdown(self, markdown: str) -> str:
        """
        Clean and normalize markdown content

        Args:
            markdown: Raw markdown from Docling

        Returns:
            Cleaned markdown
        """
        import re

        # Remove excessive blank lines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)

        # Remove page numbers (common pattern)
        markdown = re.sub(r'^Trang \d+.*\n', '', markdown, flags=re.MULTILINE)
        markdown = re.sub(r'^Page \d+.*\n', '', markdown, flags=re.MULTILINE)

        # Normalize whitespace
        markdown = re.sub(r'[ \t]+', ' ', markdown)

        # Remove leading/trailing whitespace
        markdown = markdown.strip()

        return markdown

    def batch_process(self, file_paths: list) -> dict:
        """
        Process multiple documents

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping file_path -> markdown_content
        """
        results = {}

        for file_path in file_paths:
            try:
                markdown = self.process_document(file_path)
                results[file_path] = markdown
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results[file_path] = None

        return results


# Global processor instance (lazy loading)
_docling_processor = None


def get_docling_processor(
        use_ocr: bool = True,
        use_gpu: bool = True,
        image_scale: float = 2.0,
        force_full_page_ocr: bool = False
) -> DoclingProcessor:
    """
    Get or create global Docling processor instance

    Args:
        use_ocr: Enable OCR
        use_gpu: Use GPU if available
        image_scale: Image resolution scale
        force_full_page_ocr: Force OCR on all pages

    Returns:
        DoclingProcessor instance
    """
    global _docling_processor

    if _docling_processor is None:
        _docling_processor = DoclingProcessor(
            use_ocr=use_ocr,
            use_gpu=use_gpu,
            image_scale=image_scale,
            force_full_page_ocr=force_full_page_ocr
        )

    return _docling_processor